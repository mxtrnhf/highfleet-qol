//! Ports the Steam 1.151 firing-arc rules to Steam 1.163 (the `classic` elevation rules and the
//! `classic_kaz` flag).
//!
//! Each patch is applied only when the bytes are what was analysed:
//!
//! 1. `World_RaycastBodies` (`0x140239f30`) adds one to the height level of a candidate body whose
//!    master is missing or of type 7 (free-standing parts: swivel guns, engines, legs). Steam
//!    1.151 does not, so there such a part never blocks a gun on the same level. The `cmovz` that
//!    keeps the plain level for other bodies becomes a plain `mov`.
//! 2. The KAZ module accepts a target when the bearing is below a half-angle: 1.2 rad in 1.151,
//!    pi/4 in 1.163. Two `movss` loads in `Weapon_FlakInterceptScan` are pointed at a 1.2f.
//! 3. The KAZ is blocked when part of its own 360 sectors is covered. 1.151 looked at the rear
//!    34 degrees (sector type 3) with a 50% threshold; the 1.163 data uses type 5 (rear 90
//!    degrees) with 15%. Type 5 is only used by the KAZ, so in `UpdateEngineCovered` the type 5
//!    branch is replaced by "load 0.5 as the threshold, then take the type 3 branch".
//! 4. (`classic_kaz` only, not a 1.151 rule) Trusses (`MDL_FERMA*`) are left out of the KAZ's
//!    sector scan, so they do not count as blocking its arc. Hull blocks, the command bridge,
//!    hardpoints (`MDL_HARD*`), armor and modules still do.
//!
//! Rule 1 is `elevation_rules: classic`; rules 2 to 4 are `classic_kaz`.

use std::sync::atomic::{AtomicBool, Ordering};

use patchy::{Condition, PatchSession, Result, ReturnType, Trampoline};

use crate::config::ElevationRules;

/// `lea edx,[rax+1]; cmovz edx,eax` in `World_RaycastBodies`; only the `cmovz` is replaced.
const CANDIDATE_SITE: usize = 0x14023a100;
const CANDIDATE_ORIGINAL: [u8; 6] = [0x8d, 0x50, 0x01, 0x0f, 0x44, 0xd0];
/// `mov edx,eax; nop`.
const CANDIDATE_PATCH: [u8; 3] = [0x89, 0xc2, 0x90];

/// `MOV R10,[RCX+0x88]` in `World_RaycastBodies`, right after the prologue (RSP is 16-byte aligned
/// here and RCX still holds the shooter). Detoured to remember whether the shooter is a KAZ.
const SHOOTER_SITE: usize = 0x140239f5b;
const SHOOTER_ORIGINAL: [u8; 7] = [0x4c, 0x8b, 0x91, 0x88, 0x00, 0x00, 0x00];
const SHOOTER_RETURN: usize = 0x140239f62;
/// `MOV RCX,RBX; CALL GetPartFlagsWord` in the candidate loop; the candidate body is in RBX.
const CANDIDATE_LOOP_SITE: usize = 0x14023a120;
const CANDIDATE_LOOP_ORIGINAL: [u8; 8] = [0x48, 0x8b, 0xcb, 0xe8, 0xa8, 0xb6, 0xe0, 0xff];
/// `GetPartFlagsWord`, the target of that call.
const GET_PART_FLAGS_WORD: usize = 0x1400457d0;
/// Where the loop goes on to the next candidate (the `JNZ` target of the mask test).
const CANDIDATE_SKIP: usize = 0x14023a1bc;
/// The `TEST R15D,EAX` that follows the call.
const CANDIDATE_CONTINUE: usize = 0x14023a128;

/// `GetStats(part_index)`.
const GET_STATS: usize = 0x140281e00;
const BODY_PART_INDEX: usize = 0x188;
const STATS_CATEGORY: usize = 0xb0;
/// `m_category` of `MDL_KAZ`: bit 25.
const KAZ_CATEGORY: u32 = 1 << 25;
/// `std::string` oid of a body: data or inline buffer at `+0x190`, length at `+0x1a0`, capacity
/// at `+0x1a8`.
const BODY_OID: usize = 0x190;
const BODY_OID_LEN: usize = 0x1a0;
const BODY_OID_CAP: usize = 0x1a8;
const TRUSS_PREFIX: &[u8] = b"MDL_FERMA";

/// Set at the start of each `World_RaycastBodies` call: is the shooter a KAZ module?
static SHOOTER_IS_KAZ: AtomicBool = AtomicBool::new(false);

/// `movss xmm1,[rip+disp]` loading pi/4, in the two target loops of `Weapon_FlakInterceptScan`.
const CONE_SITES: [(usize, [u8; 8]); 2] = [
    (
        0x14002f864,
        [0xf3, 0x0f, 0x10, 0x0d, 0x40, 0x2f, 0x32, 0x00],
    ),
    (
        0x14002faf8,
        [0xf3, 0x0f, 0x10, 0x0d, 0xac, 0x2c, 0x32, 0x00],
    ),
];
/// The pi/4 the sites above load.
const PI_OVER_4_ADDRESS: usize = 0x1403527ac;
/// An existing `1.2f` in `.rdata`.
const CONE_1151_ADDRESS: usize = 0x140354758;

/// The type 5 branch of `UpdateEngineCovered`: `comiss xmm6,xmm14; jc ...; movss xmm0,[...];
/// comiss xmm0,xmm6`, ending right before the shared `jc` at `0x14001c69b`.
const COVERED_SITE: usize = 0x14001c68a;
const COVERED_ORIGINAL: [u8; 17] = [
    0x41, 0x0f, 0x2f, 0xf6, 0x72, 0x36, 0xf3, 0x0f, 0x10, 0x05, 0x34, 0x81, 0x33, 0x00, 0x0f, 0x2f,
    0xc6,
];
/// The type 3 branch the new code jumps to (`comiss xmm6,xmm12`).
const COVERED_TYPE_3: usize = 0x14001c65f;
/// A `0.5f` in `.rdata`.
const HALF_ADDRESS: usize = 0x1403527a4;

/// `movss xmm1,[rip+disp32]` for an instruction at `at` reading `target`.
fn movss_xmm1(at: usize, target: usize) -> [u8; 8] {
    let disp = (target as i64 - (at as i64 + 8)) as i32;
    let mut bytes = [0xf3, 0x0f, 0x10, 0x0d, 0, 0, 0, 0];
    bytes[4..].copy_from_slice(&disp.to_le_bytes());
    bytes
}

/// The replacement for [`COVERED_ORIGINAL`]: `movss xmm10,[0.5f]; jmp type3; nop...`.
fn covered_patch() -> [u8; 17] {
    let mut bytes = [0x90; 17];
    let disp = (HALF_ADDRESS as i64 - (COVERED_SITE as i64 + 9)) as i32;
    bytes[..5].copy_from_slice(&[0xf3, 0x44, 0x0f, 0x10, 0x15]);
    bytes[5..9].copy_from_slice(&disp.to_le_bytes());
    let rel = (COVERED_TYPE_3 as i64 - (COVERED_SITE as i64 + 11)) as i8;
    bytes[9] = 0xeb;
    bytes[10] = rel as u8;
    bytes
}

/// Called at the start of every ray scan with the shooter body in RCX.
unsafe extern "C" fn note_shooter(shooter: *const u8) {
    let get_stats: unsafe extern "C" fn(i32) -> *const u8 = std::mem::transmute(GET_STATS);
    let is_kaz = if shooter.is_null() {
        false
    } else {
        let stats = get_stats(*(shooter.add(BODY_PART_INDEX) as *const i32));
        !stats.is_null() && *(stats.add(STATS_CATEGORY) as *const u32) == KAZ_CATEGORY
    };
    SHOOTER_IS_KAZ.store(is_kaz, Ordering::Relaxed);
}

/// Whether the body's oid starts with `prefix`.
unsafe fn oid_starts_with(body: *const u8, prefix: &[u8]) -> bool {
    let length = *(body.add(BODY_OID_LEN) as *const usize);
    if length < prefix.len() {
        return false;
    }
    let capacity = *(body.add(BODY_OID_CAP) as *const usize);
    let data = if capacity < 0x10 {
        body.add(BODY_OID)
    } else {
        *(body.add(BODY_OID) as *const *const u8)
    };
    std::slice::from_raw_parts(data, prefix.len()) == prefix
}

/// Called for every candidate part of a ray: true to ignore it. Only trusses are ignored, and only
/// for a KAZ shooter.
unsafe extern "C" fn skip_candidate(candidate: *const u8) -> bool {
    SHOOTER_IS_KAZ.load(Ordering::Relaxed) && oid_starts_with(candidate, TRUSS_PREFIX)
}

unsafe fn read<const N: usize>(address: usize) -> [u8; N] {
    (address as *const [u8; N]).read_unaligned()
}

unsafe fn float_at(address: usize) -> f32 {
    (address as *const f32).read_unaligned()
}

/// Prepares the classic firing-arc patches. `rules` covers the guns, `classic_kaz` the KAZ module.
pub unsafe fn patch_elevation_rules(
    session: &mut PatchSession,
    rules: ElevationRules,
    classic_kaz: bool,
) -> Result {
    if rules.is_classic() {
        patch_guns(session)?;
    } else {
        log::info!("Elevation rules are the game's own, leaving them alone");
    }

    if classic_kaz {
        patch_kaz(session)?;
    } else {
        log::info!("KAZ rules are the game's own, leaving them alone");
    }
    Ok(())
}

/// Free-standing parts stop blocking guns on their own height level.
unsafe fn patch_guns(session: &mut PatchSession) -> Result {
    if read::<6>(CANDIDATE_SITE) == CANDIDATE_ORIGINAL {
        session.overwrite(CANDIDATE_SITE + 3, &CANDIDATE_PATCH)?;
        log::info!("Elevation rules: free-standing parts no longer block same-level guns");
    } else {
        log::error!("Elevation rules: code at {CANDIDATE_SITE:#x} is not what was expected");
    }
    Ok(())
}

/// The three 1.151 KAZ rules and the truss exemption.
unsafe fn patch_kaz(session: &mut PatchSession) -> Result {
    let cone_ok = float_at(PI_OVER_4_ADDRESS) == std::f32::consts::FRAC_PI_4
        && float_at(CONE_1151_ADDRESS) == 1.2
        && CONE_SITES
            .iter()
            .all(|&(at, original)| read::<8>(at) == original);
    if cone_ok {
        for (at, _) in CONE_SITES {
            session.overwrite(at, &movss_xmm1(at, CONE_1151_ADDRESS))?;
        }
        log::info!("Classic KAZ: target cone widened to 1.2 rad");
    } else {
        log::error!("Classic KAZ: cone code is not what was expected, leaving it alone");
    }

    if read::<17>(COVERED_SITE) == COVERED_ORIGINAL && float_at(HALF_ADDRESS) == 0.5 {
        session.overwrite(COVERED_SITE, &covered_patch())?;
        log::info!("Classic KAZ: covered test uses the 1.151 rear cone and 50% threshold");
    } else {
        log::error!("Classic KAZ: covered code is not what was expected, leaving it alone");
    }

    if read::<7>(SHOOTER_SITE) == SHOOTER_ORIGINAL
        && read::<8>(CANDIDATE_LOOP_SITE) == CANDIDATE_LOOP_ORIGINAL
    {
        let mut shooter = Trampoline::new();
        shooter.preserved_call(note_shooter as *const (), &[], &[], ReturnType::None);
        shooter.bytes(&SHOOTER_ORIGINAL);
        shooter.relative_jump(SHOOTER_RETURN);
        session.detour_trampoline(SHOOTER_SITE, SHOOTER_ORIGINAL.len(), shooter)?;

        let mut candidate = Trampoline::new();
        let skip = candidate.new_label();
        candidate.preserved_call(
            skip_candidate as *const (),
            &[0x48, 0x89, 0xD9], // MOV RCX, RBX
            &[0x84, 0xC0],       // TEST AL, AL
            ReturnType::None,
        );
        candidate.jump_if(Condition::NotEqual, skip);
        candidate.bytes(&CANDIDATE_LOOP_ORIGINAL[..3]); // MOV RCX, RBX
        candidate.absolute_call(GET_PART_FLAGS_WORD as *const ());
        candidate.relative_jump(CANDIDATE_CONTINUE);
        candidate
            .bind(skip)
            .expect("truss exemption label was bound twice");
        candidate.relative_jump(CANDIDATE_SKIP);
        session.detour_trampoline(
            CANDIDATE_LOOP_SITE,
            CANDIDATE_LOOP_ORIGINAL.len(),
            candidate,
        )?;
        log::info!("Classic KAZ: trusses no longer block the KAZ's arc");
    } else {
        log::error!("Classic KAZ: ray scan code is not what was expected, trusses keep blocking");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cone_patch_matches_hand_computed_bytes() {
        assert_eq!(
            movss_xmm1(0x14002f864, CONE_1151_ADDRESS),
            [0xf3, 0x0f, 0x10, 0x0d, 0xec, 0x4e, 0x32, 0x00]
        );
        assert_eq!(
            movss_xmm1(0x14002faf8, CONE_1151_ADDRESS),
            [0xf3, 0x0f, 0x10, 0x0d, 0x58, 0x4c, 0x32, 0x00]
        );
    }

    #[test]
    fn original_cone_displacements_point_at_pi_over_4() {
        for (at, original) in CONE_SITES {
            assert_eq!(movss_xmm1(at, PI_OVER_4_ADDRESS), original);
        }
    }

    #[test]
    fn covered_patch_is_a_load_and_a_jump_to_the_type_3_branch() {
        let bytes = covered_patch();
        // movss xmm10,[rip+disp] ends at 0x14001c693; the jump at 0x14001c693 ends at 0x14001c695.
        let disp = i32::from_le_bytes(bytes[5..9].try_into().unwrap());
        assert_eq!((0x14001c693_i64 + disp as i64) as usize, HALF_ADDRESS);
        assert_eq!(bytes[9], 0xeb);
        assert_eq!(
            0x14001c695_i64 + bytes[10] as i8 as i64,
            COVERED_TYPE_3 as i64
        );
        assert!(bytes[11..].iter().all(|&b| b == 0x90));
    }

    #[test]
    fn truss_prefix_is_read_from_inline_and_heap_strings() {
        let mut body = vec![0u8; 0x1b0];
        let inline = b"MDL_FERMA2X2_01";
        body[BODY_OID..BODY_OID + inline.len()].copy_from_slice(inline);
        body[BODY_OID_LEN..BODY_OID_LEN + 8].copy_from_slice(&(inline.len() as u64).to_le_bytes());
        body[BODY_OID_CAP..BODY_OID_CAP + 8].copy_from_slice(&15u64.to_le_bytes());
        unsafe {
            assert!(oid_starts_with(body.as_ptr(), TRUSS_PREFIX));
        }

        let heap = b"MDL_HARD2X2_01_LONG_NAME".to_vec();
        let mut other = vec![0u8; 0x1b0];
        other[BODY_OID..BODY_OID + 8].copy_from_slice(&(heap.as_ptr() as u64).to_le_bytes());
        other[BODY_OID_LEN..BODY_OID_LEN + 8].copy_from_slice(&(heap.len() as u64).to_le_bytes());
        other[BODY_OID_CAP..BODY_OID_CAP + 8].copy_from_slice(&31u64.to_le_bytes());
        unsafe {
            assert!(!oid_starts_with(other.as_ptr(), TRUSS_PREFIX));
            assert!(oid_starts_with(other.as_ptr(), b"MDL_HARD"));
        }

        let short = vec![0u8; 0x1b0];
        unsafe {
            assert!(!oid_starts_with(short.as_ptr(), TRUSS_PREFIX));
        }
    }

    #[test]
    fn candidate_loop_call_targets_get_part_flags_word() {
        let rel = i32::from_le_bytes(CANDIDATE_LOOP_ORIGINAL[4..8].try_into().unwrap());
        assert_eq!(
            (CANDIDATE_LOOP_SITE as i64 + 8 + rel as i64) as usize,
            GET_PART_FLAGS_WORD
        );
    }

    #[test]
    fn original_covered_branch_is_the_one_analysed() {
        // jc at 0x14001c68e lands on 0x14001c6c6, which is 0x36 after its end.
        assert_eq!(COVERED_ORIGINAL[4], 0x72);
        assert_eq!(
            COVERED_SITE + 4 + 2 + COVERED_ORIGINAL[5] as usize,
            0x14001c6c6
        );
    }
}
