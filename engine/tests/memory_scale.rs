//! Memory-scaling guard.
//!
//! Full-regulation coverage must live in shared static/immutable data. A battle
//! environment may only hold compact per-battle state, so the inline size of the
//! per-environment types must stay bounded no matter how many moves, abilities,
//! items or forms the regulation contains. These bounds are deliberately a bit
//! above the current sizes: they catch a whole-struct regression (embedding a
//! registry table or a copied Dex inside a battle), not incidental field growth.
use pa3_engine::state::{BattleState, PokemonState, SideState, TeamSet};

#[test]
fn per_environment_state_stays_compact() {
    let battle = std::mem::size_of::<BattleState>();
    let side = std::mem::size_of::<SideState>();
    let mon = std::mem::size_of::<PokemonState>();
    let set = std::mem::size_of::<TeamSet>();
    println!(
        "size_of: BattleState={battle} SideState={side} PokemonState={mon} TeamSet={set}"
    );
    // ~2 registers and a few dozen small fields: far below any registry table.
    assert!(battle <= 16 * 1024, "BattleState inline size grew to {battle} bytes");
    assert!(side <= 8 * 1024, "SideState inline size grew to {side} bytes");
    assert!(mon <= 2 * 1024, "PokemonState inline size grew to {mon} bytes");
    assert!(set <= 512, "TeamSet inline size grew to {set} bytes");
    // The eight teams a battle needs must not dominate the environment either.
    let teams = 2 * 6 * set;
    assert!(
        teams <= 8 * 1024,
        "the twelve submitted sets grew to {teams} inline bytes"
    );
    // 2,048 environments at the measured figure (about 230 MB per 1,024-env
    // group including buffers) leaves ample headroom on a 64 GB host; this
    // asserts the *inline* part cannot explode to registry scale.
    let cohort = 2048usize * battle;
    assert!(
        cohort <= 64 * 1024 * 1024,
        "2,048 battle states would need {cohort} inline bytes"
    );
}
