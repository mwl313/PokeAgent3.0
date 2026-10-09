//! The JSON adapter runs once when a shared immutable Dex is loaded. Battles
//! use compact numeric indices and native structs, not JSON or name lookups.
use crate::{
    EngineError, Result,
    stats::{Nature, STAT_NAMES},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

pub type Id = u16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: Id,
    pub base_stats: [u16; 6],
    pub types: Vec<Id>,
    pub abilities: Vec<Id>,
    pub fixed_gender: Option<u8>,
    pub max_hp: Option<u16>,
    pub weight_hg: u32,
    pub base_species: Id,
    pub is_mega: bool,
    /// Reference `nfe`: the species has an evolution (Eviolite predicate).
    pub nfe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Category {
    Physical,
    Special,
    Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    Normal,
    AdjacentFoe,
    AdjacentAlly,
    AdjacentAllyOrSelf,
    Any,
    RandomNormal,
    SelfOnly,
    AllAdjacent,
    AllAdjacentFoes,
    All,
    AllySide,
    FoeSide,
    AllyTeam,
    Allies,
    Scripted,
}

impl Target {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "normal" => Self::Normal,
            "adjacentFoe" => Self::AdjacentFoe,
            "adjacentAlly" => Self::AdjacentAlly,
            "adjacentAllyOrSelf" => Self::AdjacentAllyOrSelf,
            "any" => Self::Any,
            "randomNormal" => Self::RandomNormal,
            "self" => Self::SelfOnly,
            "allAdjacent" => Self::AllAdjacent,
            "allAdjacentFoes" => Self::AllAdjacentFoes,
            "all" => Self::All,
            "allySide" => Self::AllySide,
            "foeSide" => Self::FoeSide,
            "allyTeam" => Self::AllyTeam,
            "allies" => Self::Allies,
            "scripted" => Self::Scripted,
            _ => return Err(EngineError::AssetMismatch(format!("unknown target {s}"))),
        })
    }

    pub fn chooses_target(self) -> bool {
        matches!(
            self,
            Self::Normal
                | Self::AdjacentFoe
                | Self::AdjacentAlly
                | Self::AdjacentAllyOrSelf
                | Self::Any
        )
    }

    pub fn valid_location(self, own_slot: u8, loc: i8) -> bool {
        if loc == 0 {
            // The engine's action mask requires an explicit location for every
            // move that chooses a target; the reference protocol additionally
            // accepts a location-less choice and resolves it randomly (the
            // held-out non-Ghost Curse scene records that difference).
            return !self.chooses_target();
        }
        if !self.chooses_target() || own_slot > 1 || !(-2..=2).contains(&loc) {
            return false;
        }
        let is_self = loc == -(own_slot as i8 + 1);
        match self {
            Self::Normal | Self::Any => !is_self,
            Self::AdjacentFoe => loc > 0,
            Self::AdjacentAlly => loc < 0 && !is_self,
            Self::AdjacentAllyOrSelf => loc < 0,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Move {
    pub id: Id,
    pub move_type: Id,
    pub category: Category,
    pub target: Target,
    pub power: u16,
    pub accuracy: Option<u8>,
    pub pp: u8,
    pub priority: i8,
    pub contact: bool,
    pub protect: bool,
    pub sound: bool,
    /// `flags.reflectable`: the move can be reflected by Magic Bounce. The
    /// reference checks this flag before any other bounce predicate.
    pub reflectable: bool,
    /// `flags.minimize`: while the target carries the `minimize` volatile this
    /// move skips the accuracy roll (`onAccuracy` returns true) and deals
    /// doubled damage (`onSourceModifyDamage`).
    pub minimize: bool,
    /// `flags.heal`: the move recovers HP, so Heal Block disables it in the
    /// request and refuses it when it was committed before the volatile landed.
    pub heal: bool,
    /// `flags.bypasssub`: the action ignores the target's Substitute.
    pub bypass_sub: bool,
    pub bullet: bool,
    pub powder: bool,
    pub pulse: bool,
    pub punch: bool,
    pub slicing: bool,
    pub bite: bool,
    /// `flags.noparentalbond`: Parental Bond never adds its second hit.
    pub no_parental_bond: bool,
    pub no_pp_boosts: bool,
    pub crit_ratio: u8,
    pub hit: crate::effects::HitEffect,
    pub self_effect: Option<crate::effects::HitEffect>,
    pub secondaries: Vec<crate::effects::SecondaryEffect>,
    pub ignore_immunity: bool,
    /// `ignoreAbility`: the move ignores the target's (breakable) ability, the
    /// same flag Mold Breaker / Teravolt / Turboblaze add at `onModifyMove`.
    pub ignore_ability: bool,
    /// `flags.nosleeptalk`: excluded from Sleep Talk's candidate list.
    pub no_sleep_talk: bool,
    /// `flags.cantusetwice`: after a successful use the move is disabled in
    /// the holder's next request while it is still the last used move.
    pub cant_use_twice: bool,
    /// `flags.mustpressure`: the move charges Pressure PP from every foe, not
    /// only its apparent targets (Imprison and the side hazards declare it).
    pub must_pressure: bool,
    /// `sleepUsable`: the move stays selectable (and is not refused) while the
    /// user is asleep; only Sleep Talk and Snore declare it in the pinned data.
    pub sleep_usable: bool,
    /// `callsMove`: the move invokes another move through `useMove`.
    pub calls_move: bool,
    /// Cold reference metadata; action callbacks must never mutate shared Dex.
    pub tracks_target: bool,
    pub conversion_excluded: bool,
    pub normalize_excluded: bool,
    pub is_z: bool,
    pub is_max: bool,
    pub smart_target: bool,
    pub pledge_combo: bool,
    pub defrost: bool,
    pub thaws_target: bool,
    /// Protect-family contact punishment, executed by the blocking volatile.
    pub protect_punish: crate::effects::ProtectPunish,
    pub recoil: Option<[u16; 2]>,
    /// `mindBlownRecoil: true` moves pay half the user's maximum HP after a
    /// connecting hit, with the move itself as the damage source, so Magic
    /// Guard does not refuse it and Rock Head does not block it.
    pub mind_blown_recoil: bool,
    /// `hasCrashDamage: true`: a failed move pays half the user's base
    /// maximum HP as crash damage (High Jump Kick family).
    pub has_crash_damage: bool,
    pub drain: Option<[u16; 2]>,
    pub side_condition: Id,
    /// `slotCondition`: a condition attached to the user's *slot* (Revival
    /// Blessing), stored in the side's per-slot condition map.
    pub slot_condition: Id,
    pub weather: Id,
    pub terrain: Id,
    /// `willCrit: true` moves always land a critical hit without a crit draw.
    pub will_crit: bool,
    /// `ignoreDefensive` moves treat the defender's defense stages as zero.
    pub ignore_defensive: bool,
    /// Some moves carry a permanent Sheer Force marker without declaring
    /// secondaries (Electro Shot in the pinned data).
    pub sheer_force_boosted: bool,
    /// `ignoreEvasion` moves treat the target's evasion stages as zero.
    pub ignore_evasion: bool,
    /// One-hit KO moves. `Some(0)` is plain `ohko: true`; `Some(type)` is a
    /// typed OHKO such as Sheer Cold's `ohko: 'Ice'`.
    pub ohko: Option<Id>,
    /// Fixed damage sources resolved exactly as the reference does.
    pub fixed_damage: Option<FixedDamage>,
    /// Reference `selfdestruct` lifecycle.
    pub self_destruct: SelfDestructMode,
    /// Ported `basePowerCallback` formula, if any.
    pub bp_callback: Option<crate::effects::BasePowerKind>,
    /// Ported action-local callbacks (see `crate::effects::hook`).
    pub hooks: u64,
    /// `overrideOffensiveStat` / `overrideDefensiveStat` as `stats` indices.
    pub override_offensive_stat: Option<u8>,
    pub override_defensive_stat: Option<u8>,
    /// `overrideOffensivePokemon: 'target'` uses the defender's Attack.
    pub override_offensive_target: bool,
    /// `selfBoost` applied as a self-targeted hit after a successful move.
    pub self_boost: Option<crate::effects::HitEffect>,
    /// `breaksProtect` removes protection from every surviving target.
    pub breaks_protect: bool,
    /// `multihit` hit range. A fixed count is `[n, n]`.
    pub multihit: Option<[u8; 2]>,
    /// Reference `selfSwitch`: the user leaves the field after the move.
    pub self_switch: SelfSwitch,
    /// Reference `forceSwitch`: the target is dragged out at the end of the
    /// action (`roar`, `whirlwind`, `dragontail`, `circlethrow`).
    pub force_switch: bool,
    /// Reference `flags.failencore`: the move can never be Encored.
    pub fail_encore: bool,
    /// `flags.gravity`: the move is refused (and disabled in the request)
    /// while the Gravity pseudo-weather is up.
    pub gravity: bool,
    /// Reference `flags.futuremove`: deferred attacks (Future Sight family).
    pub future_move: bool,
    /// Ported two-turn charge callback (`onTryMove`) plus the move's own
    /// volatile condition, if the move declares one.
    pub charge: Option<crate::effects::ChargeSpec>,
    /// `priorityChargeCallback` present: the queued action inserts a
    /// `priorityChargeMove` action (order 107) that runs the callback before
    /// any move of the turn (Chilly Reception).
    pub priority_charge: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfDestructMode {
    None,
    /// The user faints before any target resolution (`explosion`).
    Always,
    /// The user faints after hitting at least one target (`finalgambit`).
    IfHit,
}

/// Reference `selfSwitch`. `CopyVolatile` (Baton Pass) and `ShedTail` carry
/// their own volatile-transfer payload and stay explicit operational errors
/// until that payload is ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfSwitch {
    None,
    /// Plain pivot (`uturn`, `voltswitch`, `flipturn`, `partingshot`,
    /// `teleport`, `chillyreception`).
    Switch,
    /// `selfSwitch: 'copyvolatile'` (Baton Pass).
    CopyVolatile,
    /// `selfSwitch: 'shedtail'` (Shed Tail).
    ShedTail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedDamage {
    /// `damage: 'level'` (Night Shade, Seismic Toss).
    Level,
    /// `damage: <number>` (Dragon Rage, Sonic Boom).
    Flat(u16),
    /// Super Fang: `max(1, floor(target hp / 2))`.
    HalfTargetHp,
    /// Endeavor: `target hp - user hp`.
    Endeavor,
    /// Final Gambit: the user's current HP, fainting the user immediately.
    UserHp,
    /// Counter / Mirror Coat: the recorded `2 * damage` of the last qualifying
    /// hit this turn, or 1 when that recorded value is zero.
    CounterStored,
    /// Metal Burst / Comeuppance: `floor(1.5 * damage)` of the last
    /// non-ally damage this turn, or 1 when that recorded value is zero.
    LastDamagedBy,
}

#[derive(Debug, Clone)]
pub struct Dex {
    pub species: Vec<Species>,
    pub moves: Vec<Move>,
    pub natures: Vec<Nature>,
    pub names: BTreeMap<String, Vec<String>>,
    pub ids: BTreeMap<String, BTreeMap<String, Id>>,
    pub asset_digest: String,
    pub type_chart: Vec<Vec<i8>>,
    pub legal_starting_species: Vec<bool>,
    pub legal_items: Vec<bool>,
    pub legal_moves_by_species: Vec<Vec<Id>>,
    pub legal_abilities_by_species: Vec<Vec<Id>>,
    pub effects: crate::effects::NativeEffects,
}

fn to_id(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

/// True when the encoded reference data anywhere contains a native-port
/// placeholder. `encode` replaces every JavaScript function with
/// `{"callback": "<owner>"}`, so a recursive scan finds nested handlers too.
fn collect_callback_keys(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if let Some(owner) = map.get("callback").and_then(|v| v.as_str()) {
                out.push(owner.to_string());
            }
            for nested in map.values() {
                collect_callback_keys(nested, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| collect_callback_keys(v, out)),
        _ => {}
    }
}

/// Declarative move fields the native generic executor handles exactly.
const HANDLED_MOVE_FIELDS: &[&str] = &[
    // Cold catalogue metadata, never read as behaviour during a battle.
    "id",
    "name",
    "fullname",
    "num",
    "gen",
    "effectType",
    "exists",
    "isNonstandard",
    // Client/AI-facing classification only; carries no battle behaviour.
    "tags",
    "sourceEffect",
    "shortDesc",
    "desc",
    "rating",
    "isViable",
    "contestType",
    "realMove",
    "zMove",
    "maxMove",
    "zMovePower",
    "isMax",
    "isZ",
    "noCopy",
    "noSketch",
    "spreadHit",
    "affectsFainted",
    "hasSheerForceBoost",
    // Executed primitive data.
    "type",
    "baseMoveType",
    "category",
    "basePower",
    "accuracy",
    "pp",
    "priority",
    "target",
    "flags",
    "critRatio",
    "ignoreAbility",
    "sleepUsable",
    "callsMove",
    "ignoreDefensive",
    "ignoreImmunity",
    "ignoreAbility",
    "ignoreNegativeOffensive",
    "ignoreOffensive",
    "ignorePositiveDefensive",
    "forceSTAB",
    "noPPBoosts",
    "secondary",
    "secondaries",
    "self",
    "boosts",
    "heal",
    "status",
    "volatileStatus",
    "drain",
    "recoil",
    "mindBlownRecoil",
    // Crash damage: High Jump Kick / Supercell Slam / Axe Kick pay half the
    // user's base maximum HP on a failed move through the same MoveFail path.
    "hasCrashDamage",
    // Only Steel Beam's `onMoveFail` is ported (via the mindBlownRecoil
    // primitive). The crash-damage moves that share the callback
    // (High Jump Kick, Supercell Slam, Axe Kick) stay blocked by their own
    // deliberately unhandled `hasCrashDamage` field.
    "onMoveFail",
    "thawsTarget",
    "willCrit",
    "basePowerCallback",
    // Ported two-turn charge callbacks (`onTryMove` + solar onBasePower).
    "onTryMove",
    "ignoreDefensive",
    "ignoreEvasion",
    "ohko",
    "selfdestruct",
    "damage",
    "damageCallback",
    // Fields whose behaviour is a ported per-move callback. The callback keys
    // below gate which exact implementations are accepted.
    "onTry",
    "onDisableMove",
    "onTryHit",
    "onAfterSubDamage",
    "onAfterMove",
    "onModifyMove",
    "onModifyType",
    "onModifyPriority",
    "onEffectiveness",
    // Move-owned callbacks with a ported native implementation. The callback
    // keys in `PORTED_MOVE_CALLBACK_KEYS` gate which exact handlers are legal.
    "onBasePower",
    "onAfterHit",
    "onTryImmunity",
    "onHit",
    "overrideOffensiveStat",
    "overrideDefensiveStat",
    "overrideOffensivePokemon",
    "selfBoost",
    "breaksProtect",
    "multihit",
    // Per-hit accuracy re-rolls (`multiaccuracy`) with the native hook.
    "multiaccuracy",
    "selfSwitch",
    "forceSwitch",
    // `smartTarget`: a multi-hit move whose hits split across the target and
    // its adjacent ally (Dragon Darts).
    "smartTarget",
    // Embedded condition declaration (e.g. `throatchop`). The condition's own
    // callbacks are still gated by `PORTED_MOVE_CALLBACK_KEYS`.
    "condition",
];

/// Reference callback keys with a native port. A move is only classified as
/// executable when every callback it declares appears here.
const PORTED_MOVE_CALLBACK_KEYS: &[&str] = &[
    "moves:superfang.damageCallback",
    "moves:endeavor.damageCallback",
    "moves:finalgambit.damageCallback",
    // Retaliation family: Counter / Mirror Coat record the last qualifying hit
    // in their one-turn volatile; Metal Burst / Comeuppance read the turn's
    // `attackedBy` record and retarget the recorded attacker.
    "moves:counter.beforeTurnCallback",
    "moves:counter.damageCallback",
    "moves:counter.onTry",
    "moves:counter.condition.onStart",
    "moves:counter.condition.onRedirectTarget",
    "moves:counter.condition.onDamagingHit",
    "moves:mirrorcoat.beforeTurnCallback",
    "moves:mirrorcoat.damageCallback",
    "moves:mirrorcoat.onTry",
    "moves:mirrorcoat.condition.onStart",
    "moves:mirrorcoat.condition.onRedirectTarget",
    "moves:mirrorcoat.condition.onDamagingHit",
    "moves:metalburst.damageCallback",
    "moves:metalburst.onTry",
    "moves:metalburst.onModifyTarget",
    "moves:comeuppance.damageCallback",
    "moves:comeuppance.onTry",
    "moves:comeuppance.onModifyTarget",
    "moves:acrobatics.basePowerCallback",
    "moves:electroball.basePowerCallback",
    "moves:eruption.basePowerCallback",
    "moves:flail.basePowerCallback",
    "moves:grassknot.basePowerCallback",
    "moves:gyroball.basePowerCallback",
    "moves:hardpress.basePowerCallback",
    "moves:heatcrash.basePowerCallback",
    "moves:heavyslam.basePowerCallback",
    "moves:hex.basePowerCallback",
    "moves:infernalparade.basePowerCallback",
    "moves:lastrespects.basePowerCallback",
    "moves:lowkick.basePowerCallback",
    // Rage Fist / Stomping Tantrum: hit-count and last-move-result formulas.
    "moves:ragefist.basePowerCallback",
    "moves:payback.basePowerCallback",
    "moves:avalanche.basePowerCallback",
    "moves:tripleaxel.basePowerCallback",
    "moves:stompingtantrum.basePowerCallback",
    "moves:powertrip.basePowerCallback",
    "moves:reversal.basePowerCallback",
    "moves:risingvoltage.basePowerCallback",
    "moves:storedpower.basePowerCallback",
    "moves:waterspout.basePowerCallback",
    // Pure Dynamax guards. The pinned Champions format has no Dynamax and no
    // Max move state, so `target.volatiles['dynamax']` can never be set; the
    // reference callback is unreachable and natively a no-op.
    "moves:grassknot.onTryHit",
    "moves:heatcrash.onTryHit",
    "moves:heavyslam.onTryHit",
    "moves:lowkick.onTryHit",
    // Ported action-local callbacks. Each key is unique to one move, so the
    // classifier still rejects any move whose declared behaviour differs.
    "moves:fakeout.onTry",
    "moves:fakeout.onDisableMove",
    "moves:suckerpunch.onTry",
    "moves:hurricane.onModifyMove",
    "moves:thunder.onModifyMove",
    "moves:blizzard.onModifyMove",
    "moves:grassyglide.onModifyPriority",
    "moves:freezedry.onEffectiveness",
    // Ported action-local callbacks of the `selfSwitch` pivot family.
    "moves:teleport.onTry",
    "moves:partingshot.onHit",
    // Item removal: the boost is gated on the same TakeItem check as the
    // removal itself, so both halves are one ported family.
    "moves:knockoff.onBasePower",
    "moves:knockoff.onAfterHit",
    // Item swap: immunity precedes accuracy; the swap itself is native.
    "moves:trick.onTryImmunity",
    "moves:trick.onHit",
    "moves:switcheroo.onTryImmunity",
    "moves:switcheroo.onHit",
    // Ability exchange: the fail gates, End/Start ordering and announcement
    // are native (`Battle#skillSwap`).
    "moves:skillswap.onHit",
    // Stat swaps: the two boost stages (or the stored Speed stats) swap
    // through a direct write, announced afterwards.
    "moves:powerswap.onHit",
    "moves:guardswap.onHit",
    "moves:speedswap.onHit",
    // Entry hazards: the side-condition start/restart/switch-in callbacks are
    // executed by the native hazard primitive.
    "moves:spikes.condition.onSideStart",
    "moves:spikes.condition.onSideRestart",
    "moves:spikes.condition.onSwitchIn",
    "moves:stealthrock.condition.onSideStart",
    "moves:stealthrock.condition.onSwitchIn",
    "moves:toxicspikes.condition.onSideStart",
    "moves:toxicspikes.condition.onSideRestart",
    "moves:toxicspikes.condition.onSwitchIn",
    "moves:stickyweb.condition.onSideStart",
    "moves:stickyweb.condition.onSwitchIn",
    // Defog's own hit callback (evasion drop, removal, terrain clear).
    "moves:defog.onHit",
    // Item moves: the after-hit steal, the destroy-on-hit and the self
    // restore are native item primitives.
    "moves:thief.onAfterHit",
    "moves:covet.onAfterHit",
    "moves:corrosivegas.onHit",
    "moves:recycle.onHit",
    // On-hit hazard setters: the AfterHit event fires for landed hits and
    // AfterSubDamage when a decoy absorbs the attack.
    "moves:ceaselessedge.onAfterHit",
    "moves:ceaselessedge.onAfterSubDamage",
    "moves:stoneaxe.onAfterHit",
    "moves:stoneaxe.onAfterSubDamage",
    // Dire Claw's Champions secondary samples one of three major statuses and
    // applies it with `trySetStatus`; the 30% chance stays declarative.
    "moves:direclaw.secondary.onHit",
    // Trapping family: the `trapped` volatile pins the holder for as long as
    // the trapper stays active.
    "moves:block.onHit",
    "moves:meanlook.onHit",
    "moves:jawlock.onHit",
    "moves:spiritshackle.secondary.onHit",
    // Aqua Ring: the self volatile's Start message and its residual heal.
    "moves:aquaring.condition.onStart",
    "moves:aquaring.condition.onResidual",
    // Spite: four PP off the target's last move.
    "moves:spite.onHit",
    // Fell Stinger: Attack +3 when the hit KOs its target.
    "moves:fellstinger.onAfterMoveSecondarySelf",
    "moves:direclaw.secondaries.0.onHit",
    // Throat Chop's 100% secondary adds the two-turn sound-lock volatile.
    "moves:throatchop.secondary.onHit",
    "moves:throatchop.secondaries.0.onHit",
    // The embedded `throatchop` condition: start/end messages, request-level
    // sound-move disabling and the priority-6 BeforeMove refusal.
    "moves:throatchop.condition.onStart",
    "moves:throatchop.condition.onEnd",
    "moves:throatchop.condition.onDisableMove",
    "moves:throatchop.condition.onBeforeMove",
    "moves:throatchop.condition.onModifyMove",
    // Expanding Force's Psychic Terrain spread conversion and 1.5x boost.
    "moves:expandingforce.onModifyMove",
    "moves:expandingforce.onBasePower",
    // Aurora Veil's snow-only Try gate plus its side condition: Light Clay
    // duration callback, the shared screen damage modifier, and the public
    // side start/end messages.
    "moves:auroraveil.onTry",
    "moves:auroraveil.condition.durationCallback",
    "moves:auroraveil.condition.onAnyModifyDamage",
    "moves:auroraveil.condition.onSideStart",
    "moves:auroraveil.condition.onSideEnd",
    // Volatile selection-lock family (Encore / Taunt / Disable / Imprison /
    // Torment). Each block is the exact pinned Champions declaration; the
    // volatile lifecycle, request-level disable pass and BeforeMove refusals
    // live in battle.rs.
    "moves:encore.condition.onStart",
    "moves:encore.condition.onDisableMove",
    "moves:encore.condition.onResidual",
    "moves:encore.condition.onEnd",
    "moves:taunt.condition.onStart",
    "moves:taunt.condition.onDisableMove",
    "moves:taunt.condition.onBeforeMove",
    "moves:taunt.condition.onEnd",
    "moves:disable.onTryHit",
    "moves:disable.condition.onStart",
    "moves:disable.condition.onDisableMove",
    "moves:disable.condition.onBeforeMove",
    "moves:disable.condition.onEnd",
    "moves:imprison.condition.onStart",
    "moves:imprison.condition.onFoeDisableMove",
    "moves:imprison.condition.onFoeBeforeMove",
    "moves:torment.condition.onStart",
    "moves:torment.condition.onDisableMove",
    "moves:torment.condition.onEnd",
    // Two-turn charge family. Each key is unique to one pinned move and is
    // resolved to an exact recipe by `charge_shape`; a move declaring any
    // other charge callback stays an explicit operational error.
    "moves:solarbeam.onTryMove",
    "moves:solarbeam.onBasePower",
    "moves:solarblade.onTryMove",
    "moves:solarblade.onBasePower",
    "moves:electroshot.onTryMove",
    "moves:meteorbeam.onTryMove",
    "moves:skullbash.onTryMove",
    "moves:bounce.onTryMove",
    "moves:bounce.condition.onInvulnerability",
    "moves:bounce.condition.onSourceBasePower",
    "moves:dig.onTryMove",
    "moves:dig.condition.onImmunity",
    "moves:dig.condition.onInvulnerability",
    "moves:dig.condition.onSourceModifyDamage",
    "moves:dive.onTryMove",
    "moves:dive.condition.onImmunity",
    "moves:dive.condition.onInvulnerability",
    "moves:dive.condition.onSourceModifyDamage",
    "moves:fly.onTryMove",
    "moves:fly.condition.onInvulnerability",
    "moves:fly.condition.onSourceModifyDamage",
    "moves:freezeshock.onTryMove",
    "moves:geomancy.onTryMove",
    "moves:iceburn.onTryMove",
    "moves:phantomforce.onTryMove",
    "moves:razorwind.onTryMove",
    "moves:shadowforce.onTryMove",
    "moves:skyattack.onTryMove",
    // Roost: heal + one-turn Flying removal (`onStart` message, `onType`).
    "moves:roost.condition.onStart",
    "moves:roost.condition.onType",
    // Yawn: status/immunity gate plus the two-turn sleep countdown.
    "moves:yawn.onTryHit",
    "moves:yawn.condition.onStart",
    "moves:yawn.condition.onEnd",
    // Glaive Rush: the self volatile's accuracy, damage and cleanup hooks.
    "moves:glaiverush.condition.onStart",
    "moves:glaiverush.condition.onAccuracy",
    "moves:glaiverush.condition.onSourceModifyDamage",
    "moves:glaiverush.condition.onBeforeMove",
    // Minimize: the evasion volatile's restart refusal, accuracy bypass and
    // doubled damage against `flags.minimize` moves.
    "moves:minimize.condition.onRestart",
    "moves:minimize.condition.onAccuracy",
    "moves:minimize.condition.onSourceModifyDamage",
    // Clangorous Soul: HP gate, five-stat self boost and HP payment.
    "moves:clangoroussoul.onTry",
    "moves:clangoroussoul.onTryHit",
    "moves:clangoroussoul.onHit",
    // Perish Song: the field-wide countdown application and its expiry.
    "moves:perishsong.onHitField",
    "moves:perishsong.condition.onEnd",
    "moves:perishsong.condition.onResidual",
    // Leech Seed: the Grass-type immunity gate and the residual drain.
    "moves:leechseed.onTryImmunity",
    "moves:leechseed.condition.onStart",
    "moves:leechseed.condition.onResidual",
    // In-battle type changes.
    "moves:soak.onHit",
    "moves:doubleshock.onTryMove",
    "moves:doubleshock.self.onHit",
    // First Impression: the first-action gate and its request disable.
    "moves:firstimpression.onTry",
    "moves:firstimpression.onDisableMove",
    // After You: the queue reprioritisation hit effect.
    "moves:afteryou.onHit",
    // Quash: rewrites the target's queued move action order.
    "moves:quash.onHit",
    // Upper Hand: reads the target's queued move to gate itself.
    "moves:upperhand.onTry",
    // Round: the queued-action chain and its doubled base power.
    "moves:round.onTry",
    "moves:round.basePowerCallback",
    // Heal Pulse: the pulse-scaled targeted heal.
    "moves:healpulse.onHit",
    // Pain Split: the averaged HP transfer.
    "moves:painsplit.onHit",
    // Endeavor: the user-must-be-lower immunity gate; the fixed damage itself
    // runs through the ported `FixedDamage::Endeavor` primitive.
    "moves:endeavor.onTryImmunity",
    // Bug Bite / Pluck: the berry steal-and-eat hit effect.
    "moves:bugbite.onHit",
    "moves:pluck.onHit",
    // Haze and Psych Up: field-wide and copied boost stages.
    "moves:haze.onHitField",
    "moves:psychup.onHit",
    // Poltergeist: the held-item gate and its public item reveal.
    // Sleep Talk: the asleep gate and the called-move sample.
    "moves:sleeptalk.onTry",
    "moves:sleeptalk.onHit",
    // Rest: the three fail gates and the self sleep-plus-heal hit effect.
    "moves:rest.onTry",
    "moves:rest.onHit",
    // Snore: the asleep-only gate (its flinch secondary is data-driven).
    "moves:snore.onTry",
    // Charge: the condition lifecycle (start/restart/end messages, the
    // Electric base-power doubling, and the after-move/aborted consumption).
    "moves:charge.condition.onStart",
    "moves:charge.condition.onRestart",
    "moves:charge.condition.onBasePower",
    "moves:charge.condition.onMoveAborted",
    "moves:charge.condition.onAfterMove",
    "moves:charge.condition.onEnd",
    // Safeguard: the side condition's start/end and its status/volatile gates.
    "moves:safeguard.condition.onSideStart",
    "moves:safeguard.condition.onSideEnd",
    "moves:safeguard.condition.onSetStatus",
    "moves:safeguard.condition.onTryAddVolatile",
    // Smack Down: the marker's start/restart gates.
    "moves:smackdown.condition.onStart",
    "moves:smackdown.condition.onRestart",
    // No Retreat: the self-marker Try gate and its condition lifecycle.
    "moves:noretreat.onTry",
    "moves:noretreat.condition.onStart",
    "moves:noretreat.condition.onTrapPokemon",
    // Rapid-Spin-family after-hit payloads (terrain clear, hazard/seed shed).
    "moves:icespinner.onAfterHit",
    "moves:icespinner.onAfterSubDamage",
    "moves:mortalspin.onAfterHit",
    "moves:mortalspin.onAfterSubDamage",
    // Rapid Spin shares Mortal Spin's shed payload.
    "moves:rapidspin.onAfterHit",
    "moves:rapidspin.onAfterSubDamage",
    // Hazard sweep: Tidy Up's self-targeted cleanup and Court Change's side
    // condition swap.
    "moves:tidyup.onHit",
    "moves:courtchange.onHitField",
    // Wish / Healing Wish slot conditions and Heal Bell's team cure.
    "moves:wish.condition.onStart",
    "moves:wish.condition.onResidual",
    "moves:wish.condition.onEnd",
    "moves:healingwish.onTryHit",
    "moves:healingwish.condition.onSwitchIn",
    "moves:healingwish.condition.onSwap",
    "moves:healbell.onHit",
    // Swap/split family: the two Attack/Defense swap volatiles, the two stat
    // averaging moves and Magnetic Flux's side-wide boost.
    "moves:powertrick.condition.onStart",
    "moves:powertrick.condition.onCopy",
    "moves:powertrick.condition.onEnd",
    "moves:powertrick.condition.onRestart",
    "moves:powershift.condition.onStart",
    "moves:powershift.condition.onCopy",
    "moves:powershift.condition.onEnd",
    "moves:powershift.condition.onRestart",
    "moves:powersplit.onHit",
    "moves:guardsplit.onHit",
    "moves:magneticflux.onHitSide",
    // Trap/type family: Ingrain's grounding/fixing marker and Octolock's
    // trapping residual.
    "moves:ingrain.condition.onStart",
    "moves:ingrain.condition.onResidual",
    "moves:ingrain.condition.onTrapPokemon",
    "moves:ingrain.condition.onDragOut",
    "moves:octolock.onTryImmunity",
    "moves:octolock.condition.onStart",
    "moves:octolock.condition.onResidual",
    "moves:octolock.condition.onTrapPokemon",
    // Priority-charge pair: Focus Punch's focus marker and Beak Blast's
    // contact burn.
    "moves:focuspunch.priorityChargeCallback",
    "moves:focuspunch.beforeMoveCallback",
    "moves:focuspunch.condition.onStart",
    "moves:focuspunch.condition.onHit",
    "moves:focuspunch.condition.onTryAddVolatile",
    "moves:beakblast.priorityChargeCallback",
    "moves:beakblast.condition.onStart",
    "moves:beakblast.condition.onHit",
    "moves:beakblast.onAfterMove",
    // Ability-transfer moves: the gates and the setAbility payloads.
    "moves:entrainment.onTryHit",
    "moves:entrainment.onHit",
    "moves:roleplay.onTryHit",
    "moves:roleplay.onHit",
    "moves:simplebeam.onTryHit",
    "moves:simplebeam.onHit",
    // Magic Powder: the pure-Psychic type overwrite.
    "moves:magicpowder.onHit",
    // Eerie Spell: the three-PP drain on the target's last move.
    "moves:eeriespell.secondary.onHit",
    "moves:eeriespell.secondaries.0.onHit",
    // Water Shuriken: the base-power callback degenerates to the declared
    // power in the pinned regulation (Greninja-Ash + Battle Bond is not a legal
    // state and remains an explicit error).
    "moves:watershuriken.basePowerCallback",
    // Tri Attack: the sampled-status secondary, executed by the native hook.
    "moves:triattack.secondary.onHit",
    "moves:triattack.secondaries.0.onHit",
    // Burn Up: the Fire-type TryMove gate and the self type strip, both
    // executed by the native hook.
    "moves:burnup.onTryMove",
    "moves:burnup.self.onHit",
    // Raging Bull: the screen shatter at TryHit and the Paldea-form type.
    "moves:ragingbull.onTryHit",
    "moves:ragingbull.onModifyType",
    // Pollen Puff: the ally-retarget payload and its Heal Block gate.
    "moves:pollenpuff.onTryHit",
    "moves:pollenpuff.onTryMove",
    "moves:pollenpuff.onHit",
    // Curse: the Ghost/non-Ghost split and its drain condition.
    "moves:curse.onModifyMove",
    "moves:curse.onTryHit",
    "moves:curse.onHit",
    "moves:curse.condition.onStart",
    "moves:curse.condition.onResidual",
    // Baton Pass: the `canSwitch`/commanded gate and the marker that makes the
    // incoming Pokémon skip its BeforeSwitchOut event. The volatile transfer
    // itself is the `selfSwitch: 'copyvolatile'` payload.
    "moves:batonpass.onHit",
    "moves:batonpass.self.onHit",
    // Shed Tail: the canSwitch/commanded/decoy/HP gates and its own direct
    // damage; the decoy comes from the shared `substitute` volatileStatus.
    "moves:shedtail.onTryHit",
    "moves:shedtail.onHit",
    "moves:shedtail.self.onHit",
    // Focus Energy / Dragon Cheer: the mutual-exclusion start gate and the
    // crit-ratio modifier.
    "moves:focusenergy.condition.onStart",
    "moves:focusenergy.condition.onModifyCritRatio",
    "moves:dragoncheer.condition.onStart",
    "moves:dragoncheer.condition.onModifyCritRatio",
    // Revival Blessing: the no-fainted-party gate; the slot condition plus
    // selfSwitch drive the revive request and the delayed revive action.
    "moves:revivalblessing.onTryHit",
    // Destiny Bond: the consecutive-use gate; the volatile condition's
    // onFaint / onBeforeMove / onMoveAborted handlers are behaviour-driven.
    "moves:destinybond.onPrepareHit",
    // Facade: the status-doubled base power.
    "moves:facade.onBasePower",
    // Burning Jealousy: the burn for targets whose stats were raised.
    "moves:burningjealousy.onHit",
    "moves:poltergeist.onTry",
    "moves:poltergeist.onTryHit",
    // Steel Beam: `onMoveFail` is executed by the `mindBlownRecoil` primitive
    // (half the user's maximum HP as move damage on a miss or Protect block).
    "moves:steelbeam.onMoveFail",
    "moves:highjumpkick.onMoveFail",
    "moves:supercellslam.onMoveFail",
    "moves:axekick.onMoveFail",
    // Substitute: the self-cost decoy, its fail gates and the primary-hit
    // interception condition.
    "moves:substitute.onTryHit",
    "moves:substitute.onHit",
    "moves:substitute.condition.onStart",
    "moves:substitute.condition.onTryPrimaryHit",
    "moves:substitute.condition.onEnd",
    // Strength Sap: the Attack-derived heal and the Attack drop.
    "moves:strengthsap.onHit",
    // Beat Up: the captured party list and its per-hit base power formula.
    "moves:beatup.onModifyMove",
    "moves:beatup.basePowerCallback",
    // Steel Roller: the terrain-presence Try gate plus the terrain clear on a
    // landed hit and on a hit a substitute absorbs.
    "moves:steelroller.onTry",
    "moves:steelroller.onHit",
    "moves:steelroller.onAfterSubDamage",
    // The Stockpile family. Stockpile and Swallow run through their own
    // native behaviors; Spit Up is declarative with a stockpile-derived base
    // power, a stockpile-presence Try gate and the always-running AfterMove
    // consumption.
    "moves:stockpile.onTry",
    "moves:stockpile.condition.onStart",
    "moves:stockpile.condition.onRestart",
    "moves:stockpile.condition.onEnd",
    "moves:swallow.onTry",
    "moves:swallow.onHit",
    "moves:spitup.basePowerCallback",
    "moves:spitup.onTry",
    "moves:spitup.onAfterMove",
    // Turn-history power and secondary callbacks (Assurance, Temper Flare,
    // Lash Out, Barb Barrage, Alluring Voice).
    "moves:assurance.basePowerCallback",
    "moves:temperflare.basePowerCallback",
    "moves:lashout.onBasePower",
    "moves:barbbarrage.onBasePower",
    // `moves:venoshock.onBasePower`: the same doubling against a poisoned
    // target.
    "moves:venoshock.onBasePower",
    // Growth: the sun branch upgrades the declared self boost to +2/+2.
    "moves:growth.onModifyMove",
    // Fickle Beam: the 30% BasePower doubling roll.
    "moves:ficklebeam.onBasePower",
    // Last Resort: fails until every other move slot has been used.
    "moves:lastresort.onTry",
    // `moves:topsyturvy.onHit`: invert the target's boost stages.
    "moves:topsyturvy.onHit",
    // `moves:clearsmog.onHit`: reset the target's boost stages after the hit.
    "moves:clearsmog.onHit",
    "moves:alluringvoice.secondary.onHit",
    "moves:alluringvoice.secondaries.0.onHit",
];

/// Ported action-local callbacks, keyed by move id. Every entry must have its
/// reference callbacks listed in `PORTED_MOVE_CALLBACK_KEYS` so the classifier
/// rejects any move whose declared behaviour differs from the native port.
/// Reference move flags whose behaviour the native engine implements, or that
/// can only matter through an effect that is itself still an explicit
/// operational error. A move carrying any other flag (for example
/// `cantusetwice`, whose request-level disable is not ported) stays
/// `Unimplemented` instead of silently behaving as a plain attack.
const HANDLED_MOVE_FLAGS: &[&str] = &[
    // Implemented flag-driven behaviour.
    "contact",
    "protect",
    "powder",
    "punch",
    "bite",
    "slicing",
    "pulse",
    "defrost",
    // `charge` is executed by the ported two-turn recipe, and `recharge` by the
    // locked "Recharge" request entry plus the mustrecharge BeforeMove gate.
    "charge",
    "recharge",
    // `cantusetwice` disables the move in the holder's next request while it
    // is still the last used move (Gigaton Hammer).
    "cantusetwice",
    // `gravity` moves are refused (and disabled in the request) while the
    // Gravity pseudo-weather is up.
    "gravity",
    // Cold / AI-facing flags.
    "allyanim",
    "distance",
    "failcopycat",
    "failencore",
    "failinstruct",
    "failmefirst",
    "failmimic",
    "futuremove",
    "metronome",
    "mirror",
    "noassist",
    "nosketch",
    "nosleeptalk",
    "reflectable",
    "snatch",
    "sound",
    // `heal` is consumed by the ported Heal Block volatile: Heal Block
    // disables the holder's heal-flag moves in the request and refuses one
    // that was already committed.
    "heal",
    // Flags whose only consumers (Bulletproof, Dancer, Wind Rider, Gravity,
    // Substitute, Minimize, Knock Off's item gate) remain explicit errors, so
    // they cannot change an implemented mechanic yet.
    "bullet",
    "bypasssub",
    "dance",
    "gravity",
    "minimize",
    "nonsky",
    "noparentalbond",
    "wind",
    // `mustpressure` is only read by the Pressure ability, which is itself an
    // explicit operational error until ported, so it cannot change an
    // implemented mechanic.
    "mustpressure",
];

fn move_hooks(id: &str) -> u64 {
    use crate::effects::hook;
    match id {
        "fakeout" => hook::FAKE_OUT_FIRST_TURN,
        "suckerpunch" => hook::SUCKER_PUNCH,
        "hurricane" | "thunder" => hook::ACCURACY_RAIN_SUN,
        "blizzard" => hook::ACCURACY_SNOW,
        "grassyglide" => hook::PRIORITY_GRASSY_GLIDE,
        "freezedry" => hook::FREEZE_DRY,
        "knockoff" => hook::KNOCK_OFF,
        "teleport" => hook::TELEPORT,
        "partingshot" => hook::PARTING_SHOT,
        "direclaw" => hook::DIRE_CLAW,
        "throatchop" => hook::THROAT_CHOP,
        "expandingforce" => hook::EXPANDING_FORCE,
        "auroraveil" => hook::AURORA_VEIL,
        "disable" => hook::DISABLE_TARGET_GATE,
        "clangoroussoul" => hook::CLANGOROUS_SOUL,
        "populationbomb" | "tripleaxel" => hook::MULTI_ACCURACY,
        "soak" => hook::SOAK,
        "firstimpression" => hook::FIRST_IMPRESSION,
        "afteryou" => hook::AFTER_YOU,
        "haze" => hook::HAZE,
        "psychup" => hook::PSYCH_UP,
        "poltergeist" => hook::POLTERGEIST,
        "strengthsap" => hook::STRENGTH_SAP,
        "doubleshock" => hook::DOUBLE_SHOCK,
        "beatup" => hook::BEAT_UP,
        "steelroller" => hook::STEEL_ROLLER,
        "spitup" => hook::SPIT_UP,
        "lashout" => hook::LASH_OUT,
        "barbbarrage" => hook::BARB_BARRAGE,
        "alluringvoice" => hook::ALLURING_VOICE,
        "snore" => hook::SNORE,
        "facade" => hook::FACADE,
        "burningjealousy" => hook::BURNING_JEALOUSY,
        "acupressure" => hook::ACUPRESSURE,
        "magicpowder" => hook::MAGIC_POWDER,
        "entrainment" => hook::ENTRAINMENT,
        "roleplay" => hook::ROLE_PLAY,
        "simplebeam" => hook::SIMPLE_BEAM,
        "icespinner" => hook::ICE_SPINNER,
        "mortalspin" => hook::MORTAL_SPIN,
        "rapidspin" => hook::RAPID_SPIN,
        "noretreat" => hook::NO_RETREAT,
        "eeriespell" => hook::EERIE_SPELL,
        "burnup" => hook::BURN_UP,
        "triattack" => hook::TRI_ATTACK,
        "synthesis" | "moonlight" | "morningsun" => hook::WEATHER_HEAL,
        "ragingbull" => hook::RAGING_BULL,
        "pollenpuff" => hook::POLLEN_PUFF,
        "curse" => hook::CURSE,
        "venoshock" => hook::VENOSHOCK,
        "growth" => hook::GROWTH,
        "ficklebeam" => hook::FICKLE_BEAM,
        _ => 0,
    }
}

/// Move-id allowlists of the reference `condition.onInvulnerability` recipes
/// and their doubled-damage moves. `dig`/`dive` also declare `onImmunity`
/// (sandstorm/hail chip) and double through `onSourceModifyDamage`; `bounce`
/// doubles through `onSourceBasePower`. Values are transcribed from the pinned
/// callback bodies.
fn charge_condition_recipe(
    id: &str,
) -> Option<(&'static [&'static str], &'static [&'static str], bool, bool)> {
    const FLY: &[&str] = &[
        "gust",
        "twister",
        "skyuppercut",
        "thunder",
        "hurricane",
        "smackdown",
        "thousandarrows",
    ];
    const FLY_DOUBLE: &[&str] = &["gust", "twister"];
    const DIG: &[&str] = &["earthquake", "magnitude"];
    const DIVE: &[&str] = &["surf", "whirlpool"];
    // (invulnerability exceptions, doubled moves, weather immunity, doubles
    // through onSourceBasePower instead of onSourceModifyDamage)
    Some(match id {
        "fly" => (FLY, FLY_DOUBLE, false, false),
        "bounce" => (FLY, FLY_DOUBLE, false, true),
        "dig" => (DIG, DIG, true, false),
        "dive" => (DIVE, DIVE, true, false),
        _ => return None,
    })
}

/// Exact `onTryMove` recipe of a two-turn charge move. Each entry is the pinned
/// move's own callback key resolved to native behaviour; the cold classifier
/// additionally verifies the whole declared callback set, so a move whose
/// reference declaration changes stops being executable instead of silently
/// running the wrong recipe.
/// Name-keyed form of `ChargeSpec`, resolved to compact ids by the loader.
#[derive(Default)]
pub(crate) struct ChargeShape {
    pub instant_weather: &'static [&'static str],
    pub instant_weather_message: bool,
    pub prepare_boost: [i8; 7],
    pub semi_invulnerable: bool,
    pub invuln_exceptions: Vec<&'static str>,
    pub damage_double: Vec<&'static str>,
    pub power_double: Vec<&'static str>,
    pub weather_immune: bool,
    pub volatile_duration: Option<u16>,
    pub half_in_weak_weather: bool,
}

fn charge_instant_weather(id: &str) -> &'static [&'static str] {
    // `['sunnyday','desolateland']` / `['raindance','primordialsea']` in the
    // pinned bodies; the Primal weathers are not part of the regulation, so
    // only the base weather can ever be active.
    match id {
        "solarbeam" | "solarblade" => &["sunnyday"],
        "electroshot" => &["raindance"],
        _ => &[],
    }
}

fn charge_shape(id: &str, data: &Value) -> Option<ChargeShape> {
    if data["flags"]["charge"].as_u64() != Some(1) {
        return None;
    }
    let mut spec = ChargeShape {
        instant_weather: charge_instant_weather(id),
        ..Default::default()
    };
    let mut expected: Vec<String> = vec![format!("moves:{id}.onTryMove")];
    match id {
        "solarbeam" | "solarblade" => {
            expected.push(format!("moves:{id}.onBasePower"));
            // Sun skips the charge turn (read through the message-flag form of
            // `effectiveWeather`, exact for Mega Sol's activate message).
            spec.instant_weather_message = true;
            spec.half_in_weak_weather = true;
        }
        // Boost index order is [atk, def, spa, spd, spe, accuracy, evasion].
        "electroshot" => spec.prepare_boost[2] = 1,
        "meteorbeam" => spec.prepare_boost[2] = 1,
        "skullbash" => spec.prepare_boost[1] = 1,
        "bounce" | "dig" | "dive" | "fly" | "freezeshock" | "geomancy" | "iceburn"
        | "phantomforce" | "razorwind" | "shadowforce" | "skyattack" => {}
        _ => return None,
    }
    match charge_condition_recipe(id) {
        Some((exceptions, doubled, immunity, base_power)) => {
            expected.push(format!("moves:{id}.condition.onInvulnerability"));
            expected.push(format!(
                "moves:{id}.condition.{}",
                if base_power {
                    "onSourceBasePower"
                } else {
                    "onSourceModifyDamage"
                }
            ));
            if immunity {
                expected.push(format!("moves:{id}.condition.onImmunity"));
            }
            if data["condition"]["duration"].as_u64() != Some(2) {
                return None;
            }
            spec.semi_invulnerable = true;
            spec.volatile_duration = Some(2);
            spec.invuln_exceptions = exceptions.to_vec();
            spec.weather_immune = immunity;
            if base_power {
                spec.power_double = doubled.to_vec();
            } else {
                spec.damage_double = doubled.to_vec();
            }
        }
        None => {
            if let Some(condition) = data.get("condition").filter(|v| !v.is_null()) {
                // Phantom Force / Shadow Force: `onInvulnerability: false`
                // with no exception list, so nothing connects.
                if condition["duration"].as_u64() != Some(2)
                    || condition["onInvulnerability"].as_bool() != Some(false)
                {
                    return None;
                }
                spec.semi_invulnerable = true;
                spec.volatile_duration = Some(2);
            }
        }
    }
    let mut declared = Vec::new();
    collect_callback_keys(data, &mut declared);
    declared.sort();
    declared.dedup();
    expected.sort();
    if declared.len() != expected.len() || declared.iter().zip(&expected).any(|(a, b)| a != b) {
        return None;
    }
    Some(spec)
}

/// `overrideOffensiveStat`/`overrideDefensiveStat` names resolved to the
/// engine's `stats` indices. Only the five non-HP stats are legal here.
fn stat_index(name: Option<&str>) -> Option<u8> {
    Some(match name? {
        "atk" => 1,
        "def" => 2,
        "spa" => 3,
        "spd" => 4,
        "spe" => 5,
        _ => return None,
    })
}

const HANDLED_STATUSES: &[&str] = &["brn", "par", "slp", "frz", "psn", "tox"];
const HANDLED_VOLATILES: &[&str] = &[
    "flinch",
    "confusion",
    // Roost's one-turn self volatile and Yawn's sleep countdown.
    "roost",
    "yawn",
    // Glaive Rush's drawback volatile (accuracy, doubled damage, cleanup).
    "glaiverush",
    // Minimize's evasion volatile (accuracy bypass and doubled damage against
    // `flags.minimize` moves, refused restart).
    "minimize",
    // Binding moves' damage/trap volatile.
    "partiallytrapped",
    // Perish Song's three-turn countdown and Leech Seed's drain.
    "perishsong",
    "leechseed",
    "mustrecharge",
    // Focus Energy / Dragon Cheer: mutually exclusive crit-ratio volatiles.
    "focusenergy",
    "dragoncheer",
    // Charge's Electric base-power doubling volatile.
    "charge",
    // No Retreat's self-trap marker.
    "noretreat",
    // Smack Down's grounding marker.
    "smackdown",
    // Volatile selection-lock family: each id is declared by exactly one move
    // (encore, taunt, disable, imprison, torment) whose callbacks are below.
    "encore",
    "taunt",
    "disable",
    "imprison",
    "torment",
    // Substitute's HP-bearing decoy volatile.
    "substitute",
    // Psychic Noise's two-turn recovery lock. The condition also declares a
    // past-generation Heal Block move, which stays an explicit error through
    // its unported callbacks.
    "healblock",
    // Curse's Ghost drain volatile.
    "curse",
    // The Outrage / Thrash / Petal Dance / Raging Fury rampage lock.
    "lockedmove",
    // Counter / Mirror Coat's one-turn recorded-hit volatiles.
    "counter",
    "mirrorcoat",
    // Power Trick / Power Shift's Attack-Defense swap markers.
    "powertrick",
    "powershift",
    // Ingrain's grounding marker and Octolock's trapping marker.
    "ingrain",
    "octolock",
    // Focus Punch's focus marker and Beak Blast's contact-burn marker.
    "focuspunch",
    "beakblast",
];

/// Status/volatile payloads of every declared effect must already have native
/// behaviour; otherwise the whole move stays an explicit operational error.
fn effect_payload_handled(effect: &Value) -> bool {
    let Some(fields) = effect.as_object() else {
        return true;
    };
    for (key, value) in fields {
        if key == "self" {
            if !effect_payload_handled(value) {
                return false;
            }
            continue;
        }
        if !matches!(
            key.as_str(),
            "chance" | "boosts" | "status" | "volatileStatus" | "onHit"
        ) {
            return false;
        }
        if key == "status" && !value.as_str().is_some_and(|s| HANDLED_STATUSES.contains(&s)) {
            return false;
        }
        if key == "volatileStatus"
            && !value
                .as_str()
                .is_some_and(|s| HANDLED_VOLATILES.contains(&s))
        {
            return false;
        }
    }
    true
}

fn move_effects_handled(data: &Value) -> bool {
    for key in ["secondary", "self"] {
        if data.get(key).is_some_and(|v| !v.is_null()) && !effect_payload_handled(&data[key]) {
            return false;
        }
    }
    if let Some(list) = data["secondaries"].as_array()
        && !list.iter().all(effect_payload_handled)
    {
        return false;
    }
    for key in ["status", "volatileStatus"] {
        if let Some(value) = data[key].as_str() {
            let handled = if key == "status" {
                HANDLED_STATUSES.contains(&value)
            } else {
                HANDLED_VOLATILES.contains(&value)
            };
            if !handled {
                return false;
            }
        }
    }
    true
}

/// Every pinned declaration of the two `DisableMove` callbacks. `endTurn` runs
/// `runEvent('DisableMove', pokemon)`, whose handler list is speed-sorted, so
/// the collected membership is RNG-visible (a fully tied set shuffles). A new
/// declaration in the pinned data must be an explicit port work item; the
/// loader fails closed instead of silently dropping it.
const DISABLE_MOVE_CONDITIONS: &[&str] = &[
    "choicelock",
    "disable",
    "encore",
    "gravity",
    "healblock",
    "taunt",
    "throatchop",
    "torment",
];
const FOE_DISABLE_MOVE_CONDITIONS: &[&str] = &["imprison"];
const DISABLE_MOVE_ABILITIES: &[&str] = &["gorillatactics"];
const DISABLE_MOVE_ITEMS: &[&str] = &["assaultvest"];

/// Conditions declaring `noCopy`: the reference `copyVolatileFrom` refuses to
/// carry them across a Baton Pass. Pinned to the exported table so a reference
/// bump that changes the set fails closed at Dex load.
const NO_COPY_CONDITIONS: &[&str] = &[
    "attract",
    "choicelock",
    "commanded",
    "commanding",
    "counter",
    "defensecurl",
    "destinybond",
    "disable",
    "dynamax",
    "encore",
    "flashfire",
    "foresight",
    "glaiverush",
    "gmaxchistrike",
    "imprison",
    "lockon",
    "minimize",
    "miracleeye",
    "mirrorcoat",
    "nightmare",
    "protosynthesis",
    "quarkdrive",
    "saltcure",
    "smackdown",
    "spotlight",
    "stockpile",
    "syrupbomb",
    "torment",
    "trapped",
    "trapper",
    "yawn",
];

/// Conditions declaring an `onCopy` callback. None is reachable from a ported
/// effect, so a transfer stays an explicit operational error.
const COPY_CALLBACK_CONDITIONS: &[&str] = &["gastroacid", "powershift", "powertrick"];

/// Collect the condition rows whose exported declaration satisfies `wanted`
/// and assert the id set matches the native port exactly.
fn condition_id_set(
    table: &Value,
    suffix: Option<&str>,
    no_copy: bool,
    expected: &[&str],
) -> Result<Vec<Id>> {
    let mut rows: Vec<(String, Id)> = table
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| {
            let data = &row["data"];
            let wanted = match suffix {
                Some(suffix) => declares_callback(data, suffix),
                None => no_copy && data["noCopy"].as_bool() == Some(true),
            };
            wanted.then(|| {
                (
                    row["id"].as_str().unwrap().to_string(),
                    row["numeric_id"].as_u64().unwrap() as Id,
                )
            })
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    let found: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
    let mut want: Vec<String> = expected.iter().map(|name| name.to_string()).collect();
    want.sort();
    if found != want {
        return Err(EngineError::AssetMismatch(format!(
            "condition set {found:?} does not match the native port {want:?}"
        )));
    }
    Ok(rows.into_iter().map(|(_, id)| id).collect())
}

/// True when the exported declaration carries the callback anywhere: the
/// exporter replaces every function with a `{callback: "owner"}` marker.
fn declares_callback(value: &Value, suffix: &str) -> bool {
    match value {
        Value::Object(map) => {
            if let Some(owner) = map.get("callback").and_then(Value::as_str) {
                return owner.ends_with(suffix);
            }
            map.values().any(|nested| declares_callback(nested, suffix))
        }
        Value::Array(items) => items.iter().any(|nested| declares_callback(nested, suffix)),
        _ => false,
    }
}

/// Reference `resolvePriority` sub-order for a collected handler, derived from
/// the effect type. `DisableMove` conditions are Pokémon volatiles, whose
/// state target is the holder — neither a Side nor the Field — so the reference
/// Condition branch resolves to 2.
fn disable_move_sub_order(effect_type: &str, name: &str) -> i32 {
    match effect_type {
        "Condition" => 2,
        "Weather" | "Format" | "Rule" | "Ruleset" => 5,
        "Ability" => match name {
            "Poison Touch" | "Perish Body" => 6,
            "Stall" => 9,
            _ => 7,
        },
        "Item" => 8,
        _ => 0,
    }
}

/// Collect one table's `DisableMove` handlers as id -> reference sub-order,
/// returning an asset error when the pinned declarations no longer match the
/// native port's explicit list.
fn disable_move_handler_ids(
    table: &Value,
    kind: &str,
    suffix: &str,
    expected: &[&str],
) -> Result<BTreeMap<Id, i32>> {
    let mut found = BTreeMap::new();
    let mut declared = Vec::new();
    for row in table.as_array().unwrap() {
        let data = &row["data"];
        if !declares_callback(data, suffix) {
            continue;
        }
        let id = row["id"].as_str().unwrap();
        declared.push(id.to_string());
        let effect_type = data["effectType"].as_str().unwrap_or("Condition");
        let name = data["name"].as_str().unwrap_or(id);
        found.insert(
            row["numeric_id"].as_u64().unwrap() as Id,
            disable_move_sub_order(effect_type, name),
        );
    }
    declared.sort();
    let mut want: Vec<String> = expected.iter().map(|name| name.to_string()).collect();
    want.sort();
    if declared != want {
        return Err(EngineError::AssetMismatch(format!(
            "{kind} {suffix} declarations {declared:?} do not match the native port {want:?}"
        )));
    }
    Ok(found)
}

/// Cold-path classification for callback-free moves. A move only becomes
/// executable when every declared field is natively handled; anything else
/// remains `Unimplemented` (an explicit operational error), never an
/// approximation.
pub(crate) fn classify_move(id: &str, data: &Value) -> crate::effects::MoveBehavior {
    use crate::effects::MoveBehavior as Behavior;
    let explicit = Behavior::compile(id);
    if explicit != Behavior::Unimplemented {
        return explicit;
    }
    // The two string `selfSwitch` payloads are ported: `copyvolatile`
    // (Baton Pass) copies the outgoing boosts and copyable volatiles, and
    // `shedtail` moves only the user's decoy to the replacement. Anything
    // else stays an explicit operational error.
    if data["selfSwitch"].as_str().is_some_and(|cause| {
        cause != "copyvolatile" && cause != "shedtail"
    }) {
        return Behavior::Unimplemented;
    }
    let mut callbacks = Vec::new();
    collect_callback_keys(data, &mut callbacks);
    if callbacks
        .iter()
        .any(|key| !PORTED_MOVE_CALLBACK_KEYS.contains(&key.as_str()))
    {
        return Behavior::Unimplemented;
    }
    if data["flags"].as_object().is_some_and(|flags| {
        flags.iter().any(|(key, value)| {
            let on = value.as_u64().unwrap_or(0) != 0 || value.as_bool() == Some(true);
            on && !HANDLED_MOVE_FLAGS.contains(&key.as_str())
        })
    }) {
        return Behavior::Unimplemented;
    }
    if !move_effects_handled(data) {
        return Behavior::Unimplemented;
    }
    let Some(fields) = data.as_object() else {
        return Behavior::Unimplemented;
    };
    if fields
        .keys()
        .any(|key| !HANDLED_MOVE_FIELDS.contains(&key.as_str()))
    {
        return Behavior::Unimplemented;
    }
    // Two-turn charge moves run through their own ported recipe; anything the
    // recipe does not match stays an explicit operational error.
    if data["flags"]["charge"].as_u64() == Some(1) && charge_shape(id, data).is_none() {
        return Behavior::Unimplemented;
    }
    if data["category"].as_str() == Some("Status") {
        Behavior::Effect
    } else {
        Behavior::Damage
    }
}

/// Cold development diagnostic: the exact reasons `classify_move` refuses a
/// move. Never called from a battle. Reason strings are stable enough to group
/// work items by cause:
///
/// * `callback:<reference key>` — declared reference callback with no port.
/// * `field:<json key>` — declarative field the generic executor cannot run.
/// * `effect:status=<x>` / `effect:volatile=<x>` / `effect:key=<x>` — embedded
///   status/volatile payload with no native lifecycle.
pub fn move_block_reasons(id: &str, data: &Value) -> Vec<String> {
    // Mirror the real classifier exactly: a move the loader accepts has no
    // blocking reasons, no matter which raw field the cause list would name.
    if classify_move(id, data) != crate::effects::MoveBehavior::Unimplemented {
        return Vec::new();
    }
    let mut out = Vec::new();
    if data["selfSwitch"].as_str().is_some() {
        out.push("field:selfSwitch=string".into());
    }
    if data["forceSwitch"].as_bool() == Some(true) {
        out.push("field:forceSwitch".into());
    }
    if data["flags"]["charge"].as_u64() == Some(1) && charge_shape(id, data).is_none() {
        out.push("field:charge".into());
    }
    let mut callbacks = Vec::new();
    collect_callback_keys(data, &mut callbacks);
    for key in callbacks {
        if !PORTED_MOVE_CALLBACK_KEYS.contains(&key.as_str()) {
            out.push(format!("callback:{key}"));
        }
    }
    if let Some(fields) = data.as_object() {
        for key in fields.keys() {
            if !HANDLED_MOVE_FIELDS.contains(&key.as_str()) {
                out.push(format!("field:{key}"));
            }
        }
    }
    let mut payload = |prefix: &str, effect: &Value| {
        let Some(fields) = effect.as_object() else {
            return;
        };
        for (key, value) in fields {
            if key == "self" {
                continue;
            }
            if !matches!(
                key.as_str(),
                "chance" | "boosts" | "status" | "volatileStatus" | "onHit"
            ) {
                out.push(format!("effect:key={key}"));
                continue;
            }
            if key == "status"
                && let Some(name) = value.as_str()
                && !HANDLED_STATUSES.contains(&name)
            {
                out.push(format!("effect:status={name}"));
            }
            if key == "volatileStatus"
                && let Some(name) = value.as_str()
                && !HANDLED_VOLATILES.contains(&name)
            {
                out.push(format!("effect:volatile={name}"));
            }
        }
        let _ = prefix;
    };
    for key in ["secondary", "self"] {
        if data.get(key).is_some_and(|v| !v.is_null()) {
            payload(key, &data[key]);
        }
    }
    if let Some(list) = data["secondaries"].as_array() {
        for entry in list {
            payload("secondaries", entry);
        }
    }
    for key in ["status", "volatileStatus"] {
        if let Some(name) = data[key].as_str() {
            let handled = if key == "status" {
                HANDLED_STATUSES.contains(&name)
            } else {
                HANDLED_VOLATILES.contains(&name)
            };
            if !handled {
                let prefix = if key == "status" { "status" } else { "volatile" };
                out.push(format!("effect:{prefix}={name}"));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Cold development diagnostic for a blocked ability: the declared reference
/// callback keys that have no native port.
pub fn ability_block_reasons(id: &str, data: &Value) -> Vec<String> {
    if crate::effects::Ability::compile(id).is_ported() {
        return Vec::new();
    }
    let mut callbacks = Vec::new();
    collect_callback_keys(data, &mut callbacks);
    callbacks.sort();
    callbacks.dedup();
    callbacks
        .into_iter()
        .map(|key| format!("callback:{key}"))
        .collect()
}

/// Cold development diagnostic for a blocked item.
pub fn item_block_reasons(id: &str, data: &Value) -> Vec<String> {
    if crate::effects::Item::compile(id) != crate::effects::Item::Unimplemented {
        return Vec::new();
    }
    let mut callbacks = Vec::new();
    collect_callback_keys(data, &mut callbacks);
    callbacks.sort();
    callbacks.dedup();
    callbacks
        .into_iter()
        .map(|key| format!("callback:{key}"))
        .collect()
}

impl Dex {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest_bytes = std::fs::read(dir.join("manifest.json"))?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        if manifest["oracle_commit"] != crate::ORACLE_COMMIT || manifest["format"] != crate::FORMAT
        {
            return Err(EngineError::AssetMismatch("rule/reference pin".into()));
        }
        let bytes = std::fs::read(dir.join("dex.json"))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if manifest["files"]["dex.json"]["sha256"] != digest {
            return Err(EngineError::AssetMismatch("dex content digest".into()));
        }
        let value: Value = serde_json::from_slice(&bytes)?;
        let tables = value["tables"]
            .as_object()
            .ok_or_else(|| EngineError::AssetMismatch("tables".into()))?;
        let mut names = BTreeMap::new();
        let mut ids = BTreeMap::new();
        for (kind, table) in tables {
            let mut ns = vec![String::new()];
            let mut ix = BTreeMap::new();
            for row in table
                .as_array()
                .ok_or_else(|| EngineError::AssetMismatch(kind.clone()))?
            {
                let id = row["id"]
                    .as_str()
                    .ok_or_else(|| EngineError::AssetMismatch("id".into()))?;
                if row["numeric_id"].as_u64() != Some(ns.len() as u64)
                    || ns.len() > u16::MAX as usize
                {
                    return Err(EngineError::AssetMismatch(
                        "non-contiguous/overflowing IDs".into(),
                    ));
                }
                ix.insert(id.into(), ns.len() as Id);
                ns.push(id.into());
            }
            names.insert(kind.clone(), ns);
            ids.insert(kind.clone(), ix);
        }
        let lookup = |kind: &str, name: &str| -> Result<Id> {
            if kind == "abilities" && name.is_empty() {
                return Ok(0);
            }
            ids[kind]
                .get(&to_id(name))
                .copied()
                .ok_or_else(|| EngineError::AssetMismatch(format!("unknown {kind}:{name}")))
        };
        let mut species = vec![Species {
            id: 0,
            base_stats: [0; 6],
            types: vec![],
            abilities: vec![],
            fixed_gender: None,
            max_hp: None,
            weight_hg: 0,
            base_species: 0,
            is_mega: false,
            nfe: false,
        }];
        for row in tables["species"].as_array().unwrap() {
            let d = &row["data"];
            let types = d["types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| lookup("types", x.as_str().unwrap()))
                .collect::<Result<_>>()?;
            let abilities = d["abilities"]
                .as_object()
                .unwrap()
                .values()
                .map(|x| lookup("abilities", x.as_str().unwrap()))
                .collect::<Result<_>>()?;
            species.push(Species {
                id: species.len() as Id,
                base_stats: STAT_NAMES.map(|k| d["baseStats"][k].as_u64().unwrap() as u16),
                types,
                abilities,
                fixed_gender: match d["gender"].as_str() {
                    Some("M") => Some(1),
                    Some("F") => Some(2),
                    Some("N") => Some(0),
                    _ => None,
                },
                max_hp: d["maxHP"].as_u64().map(|v| v as u16),
                weight_hg: d["weighthg"].as_u64().unwrap_or(0) as u32,
                base_species: lookup("species", d["baseSpecies"].as_str().unwrap())?,
                is_mega: d["isMega"].as_bool().unwrap_or(false),
                nfe: d["nfe"].as_bool().unwrap_or(false),
            });
        }
        let mut moves = vec![Move {
            id: 0,
            move_type: 0,
            category: Category::Status,
            target: Target::SelfOnly,
            power: 0,
            accuracy: None,
            pp: 0,
            priority: 0,
            contact: false,
            protect: false,
            sound: false,
            reflectable: false,
            minimize: false,
            heal: false,
            bypass_sub: false,
            bullet: false,
            powder: false,
            pulse: false,
            punch: false,
            slicing: false,
            bite: false,
            no_parental_bond: false,
            no_pp_boosts: false,
            crit_ratio: 1,
            hit: Default::default(),
            self_effect: None,
            secondaries: vec![],
            ignore_immunity: false,
            ignore_ability: false,
            no_sleep_talk: false,
            cant_use_twice: false,
            must_pressure: false,
            sleep_usable: false,
            calls_move: false,
            tracks_target: false,
            conversion_excluded: false,
            normalize_excluded: false,
            is_z: false,
            is_max: false,
            smart_target: false,
            pledge_combo: false,
            defrost: false,
            thaws_target: false,
            protect_punish: crate::effects::ProtectPunish::None,
            recoil: None,
            mind_blown_recoil: false,
            has_crash_damage: false,
            drain: None,
            side_condition: 0,
            slot_condition: 0,
            weather: 0,
            terrain: 0,
            will_crit: false,
            ignore_defensive: false,
            ignore_evasion: false,
            sheer_force_boosted: false,
            ohko: None,
            fixed_damage: None,
                self_destruct: SelfDestructMode::None,
                bp_callback: None,
                hooks: 0,
                override_offensive_stat: None,
                override_defensive_stat: None,
                override_offensive_target: false,
                self_boost: None,
                breaks_protect: false,
                multihit: None,
                self_switch: SelfSwitch::None,
                force_switch: false,
                fail_encore: false,
                gravity: false,
                future_move: false,
                priority_charge: false,
                charge: None,
            }];
        let mut native_moves = vec![crate::effects::MoveBehavior::Unimplemented];
        let mut native_move_hooks = vec![0u64];
        for row in tables["moves"].as_array().unwrap() {
            let d = &row["data"];
            let behavior = classify_move(row["id"].as_str().unwrap(), d);
            native_moves.push(behavior);
            native_move_hooks.push(move_hooks(row["id"].as_str().unwrap()));
            let implemented = behavior != crate::effects::MoveBehavior::Unimplemented;
            let effect = |data: &Value| -> Result<crate::effects::HitEffect> {
                let condition = |key: &str| -> Result<Id> {
                    match data[key].as_str() {
                        Some(name) if implemented => lookup("conditions", name),
                        Some(name) => Ok(lookup("conditions", name).unwrap_or(0)),
                        None => Ok(0),
                    }
                };
                Ok(crate::effects::HitEffect {
                    boosts: ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
                        .map(|k| data["boosts"][k].as_i64().unwrap_or(0) as i8),
                    status: condition("status")?,
                    volatile: condition("volatileStatus")?,
                    heal: data["heal"]
                        .as_array()
                        .map(|v| [v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16]),
                })
            };
            moves.push(Move {
                id: moves.len() as Id,
                move_type: lookup("types", d["type"].as_str().unwrap())?,
                category: match d["category"].as_str().unwrap() {
                    "Physical" => Category::Physical,
                    "Special" => Category::Special,
                    _ => Category::Status,
                },
                target: Target::parse(d["target"].as_str().unwrap())?,
                power: d["basePower"].as_u64().unwrap_or(0) as u16,
                accuracy: d["accuracy"].as_u64().map(|x| x as u8),
                pp: d["pp"].as_u64().unwrap() as u8,
                priority: d["priority"].as_i64().unwrap_or(0) as i8,
                contact: d["flags"]["contact"] == 1,
                protect: d["flags"]["protect"] == 1,
                sound: d["flags"]["sound"] == 1,
                reflectable: d["flags"]["reflectable"] == 1,
                minimize: d["flags"]["minimize"] == 1,
                heal: d["flags"]["heal"] == 1,
                bypass_sub: d["flags"]["bypasssub"] == 1,
                bullet: d["flags"]["bullet"] == 1,
                powder: d["flags"]["powder"] == 1,
                pulse: d["flags"]["pulse"] == 1,
                punch: d["flags"]["punch"] == 1,
                slicing: d["flags"]["slicing"] == 1,
                bite: d["flags"]["bite"] == 1,
                no_parental_bond: d["flags"]["noparentalbond"] == 1,
                no_pp_boosts: d["noPPBoosts"].as_bool().unwrap_or(false),
                crit_ratio: d["critRatio"].as_u64().unwrap_or(1) as u8,
                hit: effect(d)?,
                self_effect: d.get("self").map(effect).transpose()?,
                secondaries: d["secondaries"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|v| {
                        Ok(crate::effects::SecondaryEffect {
                            chance: v["chance"].as_u64().unwrap_or(100) as u8,
                            target: effect(v)?,
                            own: v.get("self").map(effect).transpose()?,
                        })
                    })
                    .collect::<Result<_>>()?,
                ignore_immunity: d["ignoreImmunity"].as_bool().unwrap_or(false),
                ignore_ability: d["ignoreAbility"].as_bool().unwrap_or(false),
                no_sleep_talk: d["flags"]["nosleeptalk"] == 1,
                cant_use_twice: d["flags"]["cantusetwice"] == 1,
                must_pressure: d["flags"]["mustpressure"] == 1,
                sleep_usable: d["sleepUsable"].as_bool().unwrap_or(false),
                calls_move: d["callsMove"].as_bool().unwrap_or(false),
                tracks_target: d["tracksTarget"].as_bool().unwrap_or(false),
                conversion_excluded: matches!(
                    row["id"].as_str().unwrap(),
                    "judgment"
                        | "multiattack"
                        | "naturalgift"
                        | "revelationdance"
                        | "technoblast"
                        | "terrainpulse"
                        | "weatherball"
                ),
                normalize_excluded: matches!(
                    row["id"].as_str().unwrap(),
                    "hiddenpower"
                        | "judgment"
                        | "multiattack"
                        | "naturalgift"
                        | "revelationdance"
                        | "struggle"
                        | "technoblast"
                        | "terrainpulse"
                        | "weatherball"
                ),
                is_z: d["isZ"]
                    .as_bool()
                    .unwrap_or_else(|| d["isZ"].as_str().is_some_and(|value| !value.is_empty())),
                is_max: d["isMax"]
                    .as_bool()
                    .unwrap_or_else(|| d["isMax"].as_str().is_some_and(|value| !value.is_empty())),
                smart_target: d["smartTarget"].as_bool().unwrap_or(false),
                pledge_combo: d["flags"]["pledgecombo"] == 1,
                defrost: d["flags"]["defrost"] == 1,
                thaws_target: d["thawsTarget"].as_bool().unwrap_or(false),
                protect_punish: match row["id"].as_str().unwrap() {
                    "spikyshield" => crate::effects::ProtectPunish::DamageEighthMaxHp,
                    "banefulbunker" => crate::effects::ProtectPunish::Poison,
                    "kingsshield" => crate::effects::ProtectPunish::AttackDown,
                    _ => crate::effects::ProtectPunish::None,
                },
                recoil: d["recoil"]
                    .as_array()
                    .map(|v| [v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16]),
                mind_blown_recoil: d["mindBlownRecoil"].as_bool().unwrap_or(false),
                has_crash_damage: d["hasCrashDamage"].as_bool().unwrap_or(false),
                drain: d["drain"]
                    .as_array()
                    .map(|v| [v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16]),
                terrain: match d["terrain"].as_str() {
                    Some(name) if implemented => lookup("conditions", name)?,
                    Some(name) => lookup("conditions", name).unwrap_or(0),
                    None => 0,
                },
                weather: match d["weather"].as_str() {
                    Some(name) if implemented => lookup("conditions", name)?,
                    Some(name) => lookup("conditions", name).unwrap_or(0),
                    None => 0,
                },
                side_condition: match d["sideCondition"].as_str() {
                    Some(name) if implemented => lookup("conditions", name)?,
                    Some(name) => lookup("conditions", name).unwrap_or(0),
                    None => 0,
                },
                slot_condition: match d["slotCondition"].as_str() {
                    Some(name) if implemented => lookup("conditions", name)?,
                    Some(name) => lookup("conditions", name).unwrap_or(0),
                    None => 0,
                },
                will_crit: d["willCrit"].as_bool().unwrap_or(false),
                ignore_defensive: d["ignoreDefensive"].as_bool().unwrap_or(false),
                ignore_evasion: d["ignoreEvasion"].as_bool().unwrap_or(false),
                sheer_force_boosted: d["hasSheerForceBoost"].as_bool().unwrap_or(false),
                ohko: match &d["ohko"] {
                    Value::Bool(true) => Some(0),
                    Value::String(name) if implemented => Some(lookup("types", name)?),
                    Value::String(_) => Some(0),
                    _ => None,
                },
                fixed_damage: match &d["damage"] {
                    Value::String(kind) if kind == "level" => Some(FixedDamage::Level),
                    Value::Number(value) => Some(FixedDamage::Flat(value.as_u64().unwrap() as u16)),
                    _ => match d["damageCallback"]["callback"].as_str() {
                        Some(key) if key.ends_with("superfang.damageCallback") => {
                            Some(FixedDamage::HalfTargetHp)
                        }
                        Some(key) if key.ends_with("endeavor.damageCallback") => {
                            Some(FixedDamage::Endeavor)
                        }
                        Some(key) if key.ends_with("finalgambit.damageCallback") => {
                            Some(FixedDamage::UserHp)
                        }
                        Some(key)
                            if key.ends_with("counter.damageCallback")
                                || key.ends_with("mirrorcoat.damageCallback") =>
                        {
                            Some(FixedDamage::CounterStored)
                        }
                        Some(key)
                            if key.ends_with("metalburst.damageCallback")
                                || key.ends_with("comeuppance.damageCallback") =>
                        {
                            Some(FixedDamage::LastDamagedBy)
                        }
                        _ => None,
                    },
                },
                self_destruct: match d["selfdestruct"].as_str() {
                    Some("always") => SelfDestructMode::Always,
                    Some("ifHit") => SelfDestructMode::IfHit,
                    _ => SelfDestructMode::None,
                },
                bp_callback: d["basePowerCallback"]["callback"]
                    .as_str()
                    .and_then(crate::effects::BasePowerKind::compile),
                hooks: move_hooks(row["id"].as_str().unwrap()),
                override_offensive_stat: stat_index(d["overrideOffensiveStat"].as_str()),
                override_defensive_stat: stat_index(d["overrideDefensiveStat"].as_str()),
                override_offensive_target: d["overrideOffensivePokemon"].as_str()
                    == Some("target"),
                self_boost: d.get("selfBoost").map(effect).transpose()?,
                breaks_protect: d["breaksProtect"].as_bool().unwrap_or(false),
                multihit: match &d["multihit"] {
                    Value::Number(value) => value.as_u64().map(|n| [n as u8, n as u8]),
                    Value::Array(range) if range.len() == 2 => Some([
                        range[0].as_u64().unwrap() as u8,
                        range[1].as_u64().unwrap() as u8,
                    ]),
                    _ => None,
                },
                self_switch: match &d["selfSwitch"] {
                    Value::Bool(true) => SelfSwitch::Switch,
                    Value::String(value) if value == "copyvolatile" => SelfSwitch::CopyVolatile,
                    Value::String(value) if value == "shedtail" => SelfSwitch::ShedTail,
                    _ => SelfSwitch::None,
                },
                force_switch: d["forceSwitch"].as_bool().unwrap_or(false),
                fail_encore: d["flags"]["failencore"] == 1,
                gravity: d["flags"]["gravity"] == 1,
                future_move: d["flags"]["futuremove"] == 1,
                priority_charge: d.get("priorityChargeCallback").is_some(),
                charge: match charge_shape(row["id"].as_str().unwrap(), d) {
                    Some(shape) => {
                        let resolve = |names: &[&'static str]| -> Result<Vec<Id>> {
                            names
                                .iter()
                                .map(|name| lookup("moves", name))
                                .collect()
                        };
                        Some(crate::effects::ChargeSpec {
                            instant_weather: shape
                                .instant_weather
                                .iter()
                                .map(|name| lookup("conditions", name))
                                .collect::<Result<Vec<Id>>>()?,
                            instant_weather_message: shape.instant_weather_message,
                            prepare_boost: shape.prepare_boost,
                            semi_invulnerable: shape.semi_invulnerable,
                            invuln_exceptions: resolve(&shape.invuln_exceptions)?,
                            damage_double: resolve(&shape.damage_double)?,
                            power_double: resolve(&shape.power_double)?,
                            weather_immune: shape.weather_immune,
                            volatile_duration: shape.volatile_duration,
                            half_in_weak_weather: shape.half_in_weak_weather,
                        })
                    }
                    None => None,
                },
            });
        }
        let mut natures = vec![Nature::default()];
        for row in tables["natures"].as_array().unwrap() {
            let index = |key| {
                row["data"][key]
                    .as_str()
                    .and_then(|s| STAT_NAMES.iter().position(|n| *n == s))
            };
            natures.push(Nature {
                plus: index("plus"),
                minus: index("minus"),
            });
        }
        let type_count = names["types"].len();
        let mut chart = vec![vec![0; type_count]; type_count];
        for row in tables["types"].as_array().unwrap() {
            let defender = row["numeric_id"].as_u64().unwrap() as usize;
            for (name, val) in row["data"]["damageTaken"].as_object().unwrap() {
                // The chart also contains status immunities, retained in source assets.
                if let Ok(attacker) = lookup("types", name) {
                    chart[attacker as usize][defender] = match val.as_u64() {
                        Some(1) => 1,
                        Some(2) => -1,
                        Some(3) => -127,
                        _ => 0,
                    };
                }
            }
        }
        let scope_bytes = std::fs::read(dir.join("scope.json"))?;
        if manifest["files"]["scope.json"]["sha256"]
            != format!("{:x}", Sha256::digest(&scope_bytes))
        {
            return Err(EngineError::AssetMismatch("scope digest".into()));
        }
        let scope: Value = serde_json::from_slice(&scope_bytes)?;
        let mut legal_starting_species = vec![false; species.len()];
        let mut legal_moves_by_species = vec![vec![]; species.len()];
        let mut legal_abilities_by_species = vec![vec![]; species.len()];
        for row in scope["starting_species"].as_array().unwrap() {
            let s = lookup("species", row["species"].as_str().unwrap())? as usize;
            legal_starting_species[s] = true;
            legal_moves_by_species[s] = row["learnable_moves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| lookup("moves", m.as_str().unwrap()))
                .collect::<Result<_>>()?;
            legal_abilities_by_species[s] = row["abilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| lookup("abilities", m.as_str().unwrap()))
                .collect::<Result<_>>()?;
        }
        let mut legal_items = vec![false; names["items"].len()];
        legal_items[0] = true;
        for item in scope["allowed_items"].as_array().unwrap() {
            legal_items[lookup("items", item.as_str().unwrap())? as usize] = true;
        }
        let mut native_abilities = vec![crate::effects::Ability::Unimplemented];
        native_abilities.extend(
            names["abilities"]
                .iter()
                .skip(1)
                .map(|id| crate::effects::Ability::compile(id)),
        );
        let mut mega_stones = vec![vec![]; names["items"].len()];
        let mut berry_items = vec![false; names["items"].len()];
        for row in tables["items"].as_array().unwrap() {
            if row["data"]["isBerry"].as_bool() == Some(true) {
                berry_items[row["numeric_id"].as_u64().unwrap() as usize] = true;
            }
            if let Some(mapping) = row["data"]["megaStone"].as_object() {
                mega_stones[row["numeric_id"].as_u64().unwrap() as usize] = mapping
                    .iter()
                    .map(|(base, mega)| {
                        Ok((
                            lookup("species", base)?,
                            lookup("species", mega.as_str().unwrap())?,
                        ))
                    })
                    .collect::<Result<_>>()?;
            }
        }
        let mut native_items = vec![crate::effects::Item::Unimplemented; names["items"].len()];
        native_items[0] = crate::effects::Item::None;
        for row in tables["items"].as_array().unwrap() {
            let id = row["id"].as_str().unwrap();
            native_items[row["numeric_id"].as_u64().unwrap() as usize] =
                crate::items::classify(id, &row["data"]);
        }
        // `items:<id>.fling`: the Fling move's dynamic base power and payload.
        // An unported `fling.effect` callback stays an explicit use-time error,
        // never a load-time one, because holding such an item is legal.
        let mut fling_items = vec![None; names["items"].len()];
        for row in tables["items"].as_array().unwrap() {
            let Some(data) = row["data"].get("fling") else {
                continue;
            };
            let base_power = data["basePower"].as_u64().unwrap_or(0) as u16;
            let kind = if data.get("effect").is_some() {
                match row["id"].as_str().unwrap() {
                    "mentalherb" => crate::effects::FlingKind::MentalHerb,
                    "whiteherb" => crate::effects::FlingKind::WhiteHerb,
                    _ => crate::effects::FlingKind::Unsupported,
                }
            } else if let Some(status) = data["status"].as_str() {
                crate::effects::FlingKind::Status(lookup("conditions", status)?)
            } else if let Some(volatile) = data["volatileStatus"].as_str() {
                crate::effects::FlingKind::Volatile(lookup("conditions", volatile)?)
            } else if row["data"]["isBerry"] == true {
                crate::effects::FlingKind::Berry
            } else {
                crate::effects::FlingKind::Plain
            };
            fling_items[row["numeric_id"].as_u64().unwrap() as usize] =
                Some(crate::effects::FlingSpec { base_power, kind });
        }
        // Reference `abilities:trace.onUpdate` skips every ability flagged
        // `notrace` (Trace itself among them). The flag is data, not a native
        // port gate: an excluded ability may still be unimplemented.
        let mut no_trace_abilities = vec![false; names["abilities"].len()];
        for row in tables["abilities"].as_array().unwrap() {
            if row["data"]["flags"]["notrace"] == 1 {
                no_trace_abilities[row["numeric_id"].as_u64().unwrap() as usize] = true;
            }
        }
        // `flags.breakable`: the only abilities the active move may ignore.
        let mut breakable_abilities = vec![false; names["abilities"].len()];
        for row in tables["abilities"].as_array().unwrap() {
            if row["data"]["flags"]["breakable"] == 1 {
                breakable_abilities[row["numeric_id"].as_u64().unwrap() as usize] = true;
            }
        }
        // `flags.cantsuppress`: Mummy, Wandering Spirit and the ability-swap
        // moves refuse to overwrite or exchange these abilities.
        let mut no_suppress_abilities = vec![false; names["abilities"].len()];
        for row in tables["abilities"].as_array().unwrap() {
            if row["data"]["flags"]["cantsuppress"] == 1 {
                no_suppress_abilities[row["numeric_id"].as_u64().unwrap() as usize] = true;
            }
        }
        // `flags.noentrain` / `flags.failroleplay`: the ability-transfer moves
        // refuse these holders.
        let mut no_entrain_abilities = vec![false; names["abilities"].len()];
        let mut fail_role_play_abilities = vec![false; names["abilities"].len()];
        for row in tables["abilities"].as_array().unwrap() {
            let index = row["numeric_id"].as_u64().unwrap() as usize;
            if row["data"]["flags"]["noentrain"] == 1 {
                no_entrain_abilities[index] = true;
            }
            if row["data"]["flags"]["failroleplay"] == 1 {
                fail_role_play_abilities[index] = true;
            }
        }
        // `flags.failskillswap`: Skill Swap refuses when either side's ability
        // carries the flag.
        let mut no_skill_swap_abilities = vec![false; names["abilities"].len()];
        for row in tables["abilities"].as_array().unwrap() {
            if row["data"]["flags"]["failskillswap"] == 1 {
                no_skill_swap_abilities[row["numeric_id"].as_u64().unwrap() as usize] = true;
            }
        }
        let effects = crate::effects::NativeEffects {
            abilities: native_abilities,
            fling: lookup("conditions", "fling")?,
            fling_items,
            trapped: lookup("conditions", "trapped")?,
            trapper: lookup("conditions", "trapper")?,
            aqua_ring: lookup("conditions", "aquaring")?,
            focus_punch: lookup("conditions", "focuspunch")?,
            beak_blast: lookup("conditions", "beakblast")?,
            ingrain: lookup("conditions", "ingrain")?,
            octolock: lookup("conditions", "octolock")?,
            power_trick: lookup("conditions", "powertrick")?,
            power_shift: lookup("conditions", "powershift")?,
            wish: lookup("conditions", "wish")?,
            healing_wish: lookup("conditions", "healingwish")?,
            wish_move: lookup("moves", "wish")?,
            healing_wish_move: lookup("moves", "healingwish")?,
            heal_bell_move: lookup("moves", "healbell")?,
            counter: lookup("conditions", "counter")?,
            mirrorcoat: lookup("conditions", "mirrorcoat")?,
            counter_move: lookup("moves", "counter")?,
            mirrorcoat_move: lookup("moves", "mirrorcoat")?,
            power_swap_move: lookup("moves", "powerswap")?,
            items: native_items,
            no_trace_abilities,
            breakable_abilities,
            no_suppress_abilities,
            no_entrain_abilities,
            fail_role_play_abilities,
            simple_ability: lookup("abilities", "simple")?,
            truant_ability: lookup("abilities", "truant")?,
            no_skill_swap_abilities,
            choice_lock: lookup("conditions", "choicelock")?,
            disable_move_conditions: disable_move_handler_ids(
                &tables["conditions"],
                "conditions",
                ".onDisableMove",
                DISABLE_MOVE_CONDITIONS,
            )?,
            foe_disable_move_conditions: disable_move_handler_ids(
                &tables["conditions"],
                "conditions",
                ".onFoeDisableMove",
                FOE_DISABLE_MOVE_CONDITIONS,
            )?,
            disable_move_abilities: disable_move_handler_ids(
                &tables["abilities"],
                "abilities",
                ".onDisableMove",
                DISABLE_MOVE_ABILITIES,
            )?,
            disable_move_items: disable_move_handler_ids(
                &tables["items"],
                "items",
                ".onDisableMove",
                DISABLE_MOVE_ITEMS,
            )?,
            moves: native_moves,
            move_hooks: native_move_hooks,
            fake_out: lookup("moves", "fakeout")?,
            mega_stones,
            berry_items,
            protect: lookup("conditions", "protect")?,
            stall: lookup("conditions", "stall")?,
            revival_blessing: lookup("conditions", "revivalblessing")?,
            focus_energy: lookup("conditions", "focusenergy")?,
            dragon_cheer: lookup("conditions", "dragoncheer")?,
            charge: lookup("conditions", "charge")?,
            no_retreat: lookup("conditions", "noretreat")?,
            safeguard: lookup("conditions", "safeguard")?,
            smack_down: lookup("conditions", "smackdown")?,
            charge_move: lookup("moves", "charge")?,
            spiky_shield: lookup("conditions", "spikyshield")?,
            baneful_bunker: lookup("conditions", "banefulbunker")?,
            kings_shield: lookup("conditions", "kingsshield")?,
            endure: lookup("conditions", "endure")?,
            wide_guard: lookup("conditions", "wideguard")?,
            quick_guard: lookup("conditions", "quickguard")?,
            toxic_spikes: lookup("conditions", "toxicspikes")?,
            spikes: lookup("conditions", "spikes")?,
            stealth_rock: lookup("conditions", "stealthrock")?,
            sticky_web: lookup("conditions", "stickyweb")?,
            thief_move: lookup("moves", "thief")?,
            covet_move: lookup("moves", "covet")?,
            helping_hand: lookup("conditions", "helpinghand")?,
            follow_me: lookup("conditions", "followme")?,
            rage_powder: lookup("conditions", "ragepowder")?,
            ally_switch: lookup("conditions", "allyswitch")?,
            stockpile: lookup("conditions", "stockpile")?,
            commanded: lookup("conditions", "commanded")?,
            gravity: lookup("conditions", "gravity")?,
            magic_room: lookup("conditions", "magicroom")?,
            curse: lookup("conditions", "curse")?,
            fly_move: lookup("moves", "fly")?,
            bounce_move: lookup("moves", "bounce")?,
            chilly_reception: lookup("conditions", "chillyreception")?,
            baton_pass_move: lookup("moves", "batonpass")?,
            shed_tail_move: lookup("moves", "shedtail")?,
            no_copy_conditions: condition_id_set(&tables["conditions"], None, true, NO_COPY_CONDITIONS)?,
            copy_callback_conditions: condition_id_set(
                &tables["conditions"],
                Some(".onCopy"),
                false,
                COPY_CALLBACK_CONDITIONS,
            )?,
            damp_moves: [
                lookup("moves", "explosion")?,
                lookup("moves", "mindblown")?,
                lookup("moves", "mistyexplosion")?,
                lookup("moves", "selfdestruct")?,
            ],
            must_recharge: lookup("conditions", "mustrecharge")?,
            two_turn_move: lookup("conditions", "twoturnmove")?,
            locked_move: lookup("conditions", "lockedmove")?,
            yawn: lookup("conditions", "yawn")?,
            yawn_move: lookup("moves", "yawn")?,
            first_impression: lookup("moves", "firstimpression")?,
            toxic_move: lookup("moves", "toxic")?,
            helping_hand_move: lookup("moves", "helpinghand")?,
            round: lookup("moves", "round")?,
            destiny_bond: lookup("conditions", "destinybond")?,
            destiny_bond_move: lookup("moves", "destinybond")?,
            ceaseless_edge: lookup("moves", "ceaselessedge")?,
            stone_axe: lookup("moves", "stoneaxe")?,
            roost: lookup("conditions", "roost")?,
            glaive_rush: lookup("conditions", "glaiverush")?,
            minimize: lookup("conditions", "minimize")?,
            partially_trapped: lookup("conditions", "partiallytrapped")?,
            perish_song: lookup("conditions", "perishsong")?,
            leech_seed: lookup("conditions", "leechseed")?,
            substitute: lookup("conditions", "substitute")?,
            metronome: lookup("conditions", "metronome")?,
            ability_shield: lookup("items", "abilityshield")?,
            throat_chop: lookup("conditions", "throatchop")?,
            heal_block: lookup("conditions", "healblock")?,
            mimikyu: lookup("species", "mimikyu")?,
            mimikyu_totem: lookup("species", "mimikyutotem")?,
            mimikyu_busted: lookup("species", "mimikyubusted")?,
            mimikyu_busted_totem: lookup("species", "mimikyubustedtotem")?,
            aegislash: lookup("species", "aegislash")?,
            aegislash_blade: lookup("species", "aegislashblade")?,
            tauros_paldea_combat: lookup("species", "taurospaldeacombat")?,
            tauros_paldea_blaze: lookup("species", "taurospaldeablaze")?,
            tauros_paldea_aqua: lookup("species", "taurospaldeaaqua")?,
            kings_shield_move: lookup("moves", "kingsshield")?,
            encore: lookup("conditions", "encore")?,
            taunt: lookup("conditions", "taunt")?,
            disable: lookup("conditions", "disable")?,
            imprison: lookup("conditions", "imprison")?,
            torment: lookup("conditions", "torment")?,
            me_first: lookup("moves", "mefirst")?,
            mental_herb: lookup("items", "mentalherb")?,
            aurora_veil: lookup("conditions", "auroraveil")?,
            confusion: lookup("conditions", "confusion")?,
            sticky_hold: lookup("abilities", "stickyhold")?,
            flash_fire: lookup("conditions", "flashfire")?,
            unburden: lookup("conditions", "unburden")?,
            struggle: lookup("moves", "struggle")?,
            ground: lookup("types", "ground")?,
            fire: lookup("types", "fire")?,
            normal: lookup("types", "normal")?,
            fairy: lookup("types", "fairy")?,
            water: lookup("types", "water")?,
            grass: lookup("types", "grass")?,
            bug: lookup("types", "bug")?,
            ice: lookup("types", "ice")?,
            electric: lookup("types", "electric")?,
            poison_type: lookup("types", "poison")?,
            steel: lookup("types", "steel")?,
            burn: lookup("conditions", "brn")?,
            paralysis: lookup("conditions", "par")?,
            sleep: lookup("conditions", "slp")?,
            freeze: lookup("conditions", "frz")?,
            poison: lookup("conditions", "psn")?,
            toxic: lookup("conditions", "tox")?,
            flinch: lookup("conditions", "flinch")?,
            drain: lookup("conditions", "drain")?,
            recoil: lookup("conditions", "recoil")?,
            tailwind: lookup("conditions", "tailwind")?,
            reflect: lookup("conditions", "reflect")?,
            light_screen: lookup("conditions", "lightscreen")?,
            trick_room: lookup("conditions", "trickroom")?,
            electric_terrain: lookup("conditions", "electricterrain")?,
            grassy_terrain: lookup("conditions", "grassyterrain")?,
            misty_terrain: lookup("conditions", "mistyterrain")?,
            psychic_terrain: lookup("conditions", "psychicterrain")?,
            flying: lookup("types", "flying")?,
            psychic: lookup("types", "psychic")?,
            dragon: lookup("types", "dragon")?,
            quake_moves: [
                lookup("moves", "earthquake")?,
                lookup("moves", "bulldoze")?,
                lookup("moves", "magnitude")?,
            ],
            rain: lookup("conditions", "raindance")?,
            sun: lookup("conditions", "sunnyday")?,
            sand: lookup("conditions", "sandstorm")?,
            snow: lookup("conditions", "snowscape")?,
            rock: lookup("types", "rock")?,
            dark: lookup("types", "dark")?,
            ghost: lookup("types", "ghost")?,
            fighting: lookup("types", "fighting")?,
        };
        Ok(Self {
            species,
            moves,
            natures,
            names,
            ids,
            asset_digest: format!("{:x}", Sha256::digest(&manifest_bytes)),
            type_chart: chart,
            legal_starting_species,
            legal_items,
            legal_moves_by_species,
            legal_abilities_by_species,
            effects,
        })
    }

    pub fn id(&self, kind: &str, name: &str) -> Result<Id> {
        self.ids
            .get(kind)
            .and_then(|t| t.get(&to_id(name)))
            .copied()
            .ok_or_else(|| EngineError::InvalidInput(format!("unknown {kind}:{name}")))
    }
}
