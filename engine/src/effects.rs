//! Native handler selection. Unknown entries are explicit work items, never
//! neutral effects. These enums are populated once, not by string dispatch in turns.
use crate::assets::Id;
use serde::{Deserialize, Serialize};

/// Action-local move callbacks that cannot be expressed as generic declarative
/// data. Each bit is an exact transcription of the pinned reference callback;
/// the cold classifier only enables a move when every callback it declares is
/// ported, so an unsupported variant stays an explicit operational error.
pub mod hook {
    /// `moves:fakeout.onTry` plus the Champions `onDisableMove` override.
    pub const FAKE_OUT_FIRST_TURN: u64 = 1 << 0;
    /// `moves:suckerpunch.onTry`.
    pub const SUCKER_PUNCH: u64 = 1 << 1;
    /// `moves:hurricane|thunder.onModifyMove`: rain makes the move always hit,
    /// sun sets 50% accuracy.
    pub const ACCURACY_RAIN_SUN: u64 = 1 << 2;
    /// `moves:blizzard.onModifyMove`: snow makes the move always hit.
    pub const ACCURACY_SNOW: u64 = 1 << 3;
    /// `moves:grassyglide.onModifyPriority`.
    pub const PRIORITY_GRASSY_GLIDE: u64 = 1 << 4;
    /// `moves:freezedry.onEffectiveness`.
    pub const FREEZE_DRY: u64 = 1 << 5;
    /// `moves:lowkick|grassknot.onTryHit`: the Dynamax branch cannot be reached
    /// in the pinned regulation, so a Dynamax volatile is an explicit error.
    pub const DYNAMAX_GUARD: u64 = 1 << 6;
    /// `moves:knockoff.onBasePower` plus its `onAfterHit` item removal: the
    /// 1.5x boost only applies when the item can actually be taken.
    pub const KNOCK_OFF: u64 = 1 << 7;
    /// `moves:teleport.onTry`: Teleport fails outright, before any hit step,
    /// when the user has no switchable reserve.
    pub const TELEPORT: u64 = 1 << 8;
    /// `moves:partingshot.onHit`: the pivot is cancelled when the Attack and
    /// Special Attack drop fails.
    pub const PARTING_SHOT: u64 = 1 << 9;
    /// `moves:direclaw.secondary.onHit`: the secondary samples one of
    /// poison/paralysis/sleep and applies it through `trySetStatus`.
    pub const DIRE_CLAW: u64 = 1 << 10;
    /// `moves:throatchop.secondary.onHit`: adds the two-turn volatile that
    /// disables and refuses sound moves.
    pub const THROAT_CHOP: u64 = 1 << 11;
    /// `moves:expandingforce.onModifyMove|onBasePower`: Psychic Terrain turns
    /// the move into a spread move and boosts it 1.5x for grounded users.
    pub const EXPANDING_FORCE: u64 = 1 << 12;
    /// `moves:auroraveil.onTry`: the screen only starts while snow is falling.
    pub const AURORA_VEIL: u64 = 1 << 13;
    /// `moves:disable.onTryHit`: the move fails before accuracy when the target
    /// has no recorded last move (or last used Struggle).
    pub const DISABLE_TARGET_GATE: u64 = 1 << 14;
    /// `moves:clangoroussoul.onTry|onTryHit|onHit`: the user must be above a
    /// third of its maximum HP, the five-stat self boost must change something
    /// and the move then costs a third of the user's maximum HP.
    pub const CLANGOROUS_SOUL: u64 = 1 << 15;
    /// `multiaccuracy`: the move re-rolls accuracy for every hit after the
    /// first and stops on the first miss (Population Bomb, Triple Axel).
    pub const MULTI_ACCURACY: u64 = 1 << 16;
    /// `moves:soak.onHit`: overwrite the target's types with pure Water.
    pub const SOAK: u64 = 1 << 17;
    /// `moves:doubleshock.onTryMove|self.onHit`: fails without the Electric
    /// type and removes it from the user on a landed hit.
    pub const DOUBLE_SHOCK: u64 = 1 << 18;
    /// `moves:firstimpression.onTry` plus the Champions `onDisableMove`
    /// override: only the holder's first action out may use it.
    pub const FIRST_IMPRESSION: u64 = 1 << 19;
    /// `moves:afteryou.onHit`: the ally's queued move jumps to the queue head.
    pub const AFTER_YOU: u64 = 1 << 20;
    /// `moves:haze.onHitField`: every active Pokémon's boosts are cleared.
    pub const HAZE: u64 = 1 << 21;
    /// `moves:psychup.onHit`: the user copies every boost stage of the target.
    pub const PSYCH_UP: u64 = 1 << 22;
    /// `moves:poltergeist.onTry|onTryHit`: the move fails without a held item
    /// and publicly reveals the item when it connects.
    pub const POLTERGEIST: u64 = 1 << 23;
    /// `moves:strengthsap.onHit`: heal by the target's stage-boosted Attack
    /// (no ModifyStat modifiers) and drop the target's Attack one stage.
    pub const STRENGTH_SAP: u64 = 1 << 24;
    /// `moves:beatup.onModifyMove`: the action's hit count is the number of
    /// party members that are the user or are healthy and status-free, and
    /// each hit's power comes from the next such member's set species.
    pub const BEAT_UP: u64 = 1 << 25;
    /// `moves:steelroller.onTry|onHit|onAfterSubDamage`: the move fails without
    /// an active terrain and clears it once the hit lands, including a hit a
    /// substitute absorbs.
    pub const STEEL_ROLLER: u64 = 1 << 26;
    /// `moves:spitup.onTry|onAfterMove`: the move needs the user's stockpile
    /// volatile and always removes it once the move has run, even when the
    /// hit is blocked or missed.
    pub const SPIT_UP: u64 = 1 << 27;
    /// `moves:lashout.onBasePower`: doubles while the user's stats were
    /// lowered this turn.
    pub const LASH_OUT: u64 = 1 << 28;
    /// `moves:barbbarrage.onBasePower`: doubles against a poisoned target.
    pub const BARB_BARRAGE: u64 = 1 << 29;
    /// `moves:alluringvoice.secondary.onHit`: confuses a target whose stats
    /// were raised this turn.
    pub const ALLURING_VOICE: u64 = 1 << 30;
    /// `moves:snore.onTry`: the move fails outright unless the user is
    /// asleep (or Comatose, which no in-scope ability provides).
    pub const SNORE: u64 = 1 << 31;
    /// `moves:facade.onBasePower`: doubles while the user carries a major
    /// status (burn, paralysis or poison).
    pub const FACADE: u64 = 1 << 32;
    /// `moves:burningjealousy.onHit`: burns each target whose stats were
    /// raised this turn.
    pub const BURNING_JEALOUSY: u64 = 1 << 33;
    /// `moves:acupressure.onHit`: one sampled stat below +6 rises two stages.
    pub const ACUPRESSURE: u64 = 1 << 34;
    /// `moves:magicpowder.onHit`: overwrite the target's types with pure
    /// Psychic, refusing a target that already is pure Psychic.
    pub const MAGIC_POWDER: u64 = 1 << 38;
    /// `moves:eeriespell.secondary.onHit`: deduct three PP from the target's
    /// last move.
    pub const EERIE_SPELL: u64 = 1 << 39;
    /// `moves:entrainment.onTryHit|onHit`: the target adopts the user's
    /// ability (gate: different, replaceable, `noentrain`-free).
    pub const ENTRAINMENT: u64 = 1 << 40;
    /// `moves:roleplay.onTryHit|onHit`: the user adopts the target's ability.
    pub const ROLE_PLAY: u64 = 1 << 41;
    /// `moves:simplebeam.onTryHit|onHit`: the target's ability becomes Simple.
    pub const SIMPLE_BEAM: u64 = 1 << 42;
    /// `moves:icespinner.onAfterHit|onAfterSubDamage`: a landed hit clears the
    /// active terrain.
    pub const ICE_SPINNER: u64 = 1 << 43;
    /// `moves:mortalspin.onAfterHit|onAfterSubDamage`: the user sheds Leech
    /// Seed, its own entry hazards and partial trapping.
    pub const MORTAL_SPIN: u64 = 1 << 44;
    /// `moves:rapidspin.onAfterHit|onAfterSubDamage`: the same shed, which the
    /// move shares with Mortal Spin.
    pub const RAPID_SPIN: u64 = 1 << 52;
    /// `moves:noretreat.onTry`: the move fails while its own marker volatile
    /// is present.
    pub const NO_RETREAT: u64 = 1 << 45;
    /// `moves:ragingbull.onTryHit|onModifyType`: the target side's screens are
    /// shattered before the hit resolves, and the Paldea Tauros forms use
    /// their own primary type (Combat/Fighting, Blaze/Fire, Aqua/Water).
    pub const RAGING_BULL: u64 = 1 << 46;
    /// `moves:pollenpuff.onTryHit|onTryMove|onHit`: an ally-targeted Pollen
    /// Puff becomes a zero-power heal that ignores the ally's decoy, and Heal
    /// Block on the user refuses that use outright.
    pub const POLLEN_PUFF: u64 = 1 << 47;
    /// `moves:curse.onModifyMove|onTryHit|onHit`: a Ghost user curses the foe
    /// (quarter-max-HP drain per residual, half the user's HP paid on use); any
    /// other user instead boosts itself.
    pub const CURSE: u64 = 1 << 48;
    /// `moves:venoshock.onBasePower`: doubles against a poisoned target.
    pub const VENOSHOCK: u64 = 1 << 49;
    /// `moves:growth.onModifyMove`: the sun branch upgrades the self boost.
    pub const GROWTH: u64 = 1 << 51;
    /// `moves:ficklebeam.onBasePower`: a 30% roll doubles the BasePower.
    pub const FICKLE_BEAM: u64 = 1 << 50;
    /// `moves:burnup.onTryMove|self.onHit`: fails without the Fire type and
    /// strips it from the user on a landed hit.
    pub const BURN_UP: u64 = 1 << 35;
    /// `moves:triattack.secondary.onHit`: samples burn, paralysis or freeze.
    pub const TRI_ATTACK: u64 = 1 << 36;
    /// `moves:synthesis|moonlight|morningsun.onHit` weather-scaled heal.
    pub const WEATHER_HEAL: u64 = 1 << 37;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ability {
    Unimplemented,
    Armor,
    Levitate,
    Blaze,
    Torrent,
    Overgrow,
    Swarm,
    Intimidate,
    Defiant,
    Competitive,
    ClearBody,
    InnerFocus,
    OwnTempo,
    Oblivious,
    SpeedBoost,
    ToughClaws,
    Technician,
    Adaptability,
    MegaLauncher,
    IronFist,
    Sharpness,
    StrongJaw,
    HugePower,
    Filter,
    Multiscale,
    ThickFat,
    Regenerator,
    NaturalCure,
    RockHead,
    Reckless,
    LiquidOoze,
    Infiltrator,
    Drizzle,
    Drought,
    SandStream,
    SnowWarning,
    SwiftSwim,
    Chlorophyll,
    SandRush,
    SlushRush,
    ElectricSurge,
    GrassySurge,
    MistySurge,
    PsychicSurge,
    RainDish,
    IceBody,
    SolarPower,
    CloudNine,
    AirLock,
    SandForce,
    SandVeil,
    SnowCloak,
    Overcoat,
    Hydration,
    DrySkin,
    WaterAbsorb,
    VoltAbsorb,
    EarthEater,
    SapSipper,
    MotorDrive,
    Static,
    FlashFire,
    LightningRod,
    StormDrain,
    Pixilate,
    Aerilate,
    Refrigerate,
    Galvanize,
    Normalize,
    Dragonize,
    LiquidVoice,
    HyperCutter,
    Synchronize,
    RoughSkin,
    Stamina,
    FlameBody,
    PoisonTouch,
    Prankster,
    Aftermath,
    Analytic,
    Angerpoint,
    Anticipation,
    Armortail,
    Aromaveil,
    Auraguard,
    Battlebond,
    Berserk,
    Bigpecks,
    Bulletproof,
    Cheekpouch,
    Compoundeyes,
    Contrary,
    Corrosion,
    Cudchew,
    Curiousmedicine,
    Cursedbody,
    Cutecharm,
    Damp,
    Disguise,
    Earlybird,
    Eelevate,
    Effectspore,
    Electromorphosis,
    Embodyaspectcornerstone,
    Embodyaspecthearthflame,
    Embodyaspectteal,
    Embodyaspectwellspring,
    Emergencyexit,
    Fairyaura,
    Firemane,
    Flowerveil,
    Fluffy,
    Forecast,
    Forewarn,
    Friendguard,
    Frisk,
    Furcoat,
    Galewings,
    Gluttony,
    Goodasgold,
    Gooey,
    Grasspelt,
    Guarddog,
    Gulpmissile,
    Guts,
    Harvest,
    Healer,
    Heatproof,
    Heavymetal,
    Hospitality,
    Hungerswitch,
    Hustle,
    Iceface,
    Illuminate,
    Illusion,
    Immunity,
    Imposter,
    Innardsout,
    Insomnia,
    Justified,
    Keeneye,
    Klutz,
    Leafguard,
    Libero,
    Lightmetal,
    Limber,
    Longreach,
    Magicbounce,
    Magicguard,
    Magician,
    Magmaarmor,
    Marvelscale,
    Megasol,
    Merciless,
    Mimicry,
    Minus,
    Mirrorarmor,
    Moldbreaker,
    Moody,
    Moxie,
    Mummy,
    Noguard,
    Opportunist,
    Parentalbond,
    Pickpocket,
    Pickup,
    Piercingdrill,
    Plus,
    Poisonheal,
    Poisonpoint,
    Pressure,
    Protean,
    Punkrock,
    Purifyingsalt,
    Queenlymajesty,
    Quickdraw,
    Quickfeet,
    Rattled,
    Receiver,
    Ripen,
    Rivalry,
    Runaway,
    Sandspit,
    Scrappy,
    Screencleaner,
    Seedsower,
    Shadowtag,
    Arenatrap,
    Magnetpull,
    Shedskin,
    /// `abilities:simple.onChangeBoost`: doubles every incoming boost delta.
    Simple,
    Sheerforce,
    Shielddust,
    Shieldsdown,
    Skilllink,
    Sniper,
    Soundproof,
    Spicyspray,
    Stakeout,
    Stall,
    Stalwart,
    Stancechange,
    Steadfast,
    Steelyspirit,
    Stench,
    Stickyhold,
    Sturdy,
    Suctioncups,
    Superluck,
    Supersweetsyrup,
    Supremeoverlord,
    Surgesurfer,
    Sweetveil,
    Symbiosis,
    Tangledfeet,
    Telepathy,
    Thermalexchange,
    Toxicdebris,
    Trace,
    Unaware,
    Unburden,
    Unnerve,
    Unseenfist,
    Vitalspirit,
    Wanderingspirit,
    Waterbubble,
    Weakarmor,
    Whitesmoke,
    Zerotohero,
}

impl Ability {
    pub fn compile(id: &str) -> Self {
        match id {
            "battlearmor" | "shellarmor" => Self::Armor,
            "levitate" => Self::Levitate,
            "blaze" => Self::Blaze,
            "torrent" => Self::Torrent,
            "overgrow" => Self::Overgrow,
            "swarm" => Self::Swarm,
            "intimidate" => Self::Intimidate,
            "defiant" => Self::Defiant,
            "competitive" => Self::Competitive,
            "clearbody" => Self::ClearBody,
            "innerfocus" => Self::InnerFocus,
            "owntempo" => Self::OwnTempo,
            "oblivious" => Self::Oblivious,
            "speedboost" => Self::SpeedBoost,
            "toughclaws" => Self::ToughClaws,
            "technician" => Self::Technician,
            "adaptability" => Self::Adaptability,
            "megalauncher" => Self::MegaLauncher,
            "ironfist" => Self::IronFist,
            "sharpness" => Self::Sharpness,
            "strongjaw" => Self::StrongJaw,
            "hugepower" => Self::HugePower,
            "purepower" => Self::HugePower,
            "filter" => Self::Filter,
            "solidrock" => Self::Filter,
            "multiscale" => Self::Multiscale,
            "thickfat" => Self::ThickFat,
            "regenerator" => Self::Regenerator,
            "naturalcure" => Self::NaturalCure,
            "rockhead" => Self::RockHead,
            "reckless" => Self::Reckless,
            "liquidooze" => Self::LiquidOoze,
            "infiltrator" => Self::Infiltrator,
            "drizzle" => Self::Drizzle,
            "drought" => Self::Drought,
            "sandstream" => Self::SandStream,
            "snowwarning" => Self::SnowWarning,
            "swiftswim" => Self::SwiftSwim,
            "chlorophyll" => Self::Chlorophyll,
            "sandrush" => Self::SandRush,
            "slushrush" => Self::SlushRush,
            "electricsurge" => Self::ElectricSurge,
            "grassysurge" => Self::GrassySurge,
            "mistysurge" => Self::MistySurge,
            "psychicsurge" => Self::PsychicSurge,
            "raindish" => Self::RainDish,
            "icebody" => Self::IceBody,
            "solarpower" => Self::SolarPower,
            "cloudnine" => Self::CloudNine,
            "airlock" => Self::AirLock,
            "sandforce" => Self::SandForce,
            "sandveil" => Self::SandVeil,
            "snowcloak" => Self::SnowCloak,
            "overcoat" => Self::Overcoat,
            "hydration" => Self::Hydration,
            "dryskin" => Self::DrySkin,
            "waterabsorb" => Self::WaterAbsorb,
            "voltabsorb" => Self::VoltAbsorb,
            "eartheater" => Self::EarthEater,
            "sapsipper" => Self::SapSipper,
            "motordrive" => Self::MotorDrive,
            "static" => Self::Static,
            "flashfire" => Self::FlashFire,
            "lightningrod" => Self::LightningRod,
            "stormdrain" => Self::StormDrain,
            "pixilate" => Self::Pixilate,
            "aerilate" => Self::Aerilate,
            "refrigerate" => Self::Refrigerate,
            "galvanize" => Self::Galvanize,
            "normalize" => Self::Normalize,
            "dragonize" => Self::Dragonize,
            "liquidvoice" => Self::LiquidVoice,
            "hypercutter" => Self::HyperCutter,
            "synchronize" => Self::Synchronize,
            "roughskin" => Self::RoughSkin,
            "stamina" => Self::Stamina,
            "flamebody" => Self::FlameBody,
            "poisontouch" => Self::PoisonTouch,
            "prankster" => Self::Prankster,

            "aftermath" => Self::Aftermath,
            "analytic" => Self::Analytic,
            "angerpoint" => Self::Angerpoint,
            "anticipation" => Self::Anticipation,
            "armortail" => Self::Armortail,
            "aromaveil" => Self::Aromaveil,
            "auraguard" => Self::Auraguard,
            "battlebond" => Self::Battlebond,
            "berserk" => Self::Berserk,
            "bigpecks" => Self::Bigpecks,
            "bulletproof" => Self::Bulletproof,
            "cheekpouch" => Self::Cheekpouch,
            "compoundeyes" => Self::Compoundeyes,
            "contrary" => Self::Contrary,
            "corrosion" => Self::Corrosion,
            "cudchew" => Self::Cudchew,
            "curiousmedicine" => Self::Curiousmedicine,
            "cursedbody" => Self::Cursedbody,
            "cutecharm" => Self::Cutecharm,
            "damp" => Self::Damp,
            "disguise" => Self::Disguise,
            "earlybird" => Self::Earlybird,
            "eelevate" => Self::Eelevate,
            "effectspore" => Self::Effectspore,
            "electromorphosis" => Self::Electromorphosis,
            "embodyaspectcornerstone" => Self::Embodyaspectcornerstone,
            "embodyaspecthearthflame" => Self::Embodyaspecthearthflame,
            "embodyaspectteal" => Self::Embodyaspectteal,
            "embodyaspectwellspring" => Self::Embodyaspectwellspring,
            "emergencyexit" => Self::Emergencyexit,
            "fairyaura" => Self::Fairyaura,
            "firemane" => Self::Firemane,
            "flowerveil" => Self::Flowerveil,
            "fluffy" => Self::Fluffy,
            "forecast" => Self::Forecast,
            "forewarn" => Self::Forewarn,
            "friendguard" => Self::Friendguard,
            "frisk" => Self::Frisk,
            "furcoat" => Self::Furcoat,
            "galewings" => Self::Galewings,
            "gluttony" => Self::Gluttony,
            "goodasgold" => Self::Goodasgold,
            "gooey" => Self::Gooey,
            "grasspelt" => Self::Grasspelt,
            "guarddog" => Self::Guarddog,
            "gulpmissile" => Self::Gulpmissile,
            "guts" => Self::Guts,
            "harvest" => Self::Harvest,
            "healer" => Self::Healer,
            "heatproof" => Self::Heatproof,
            "heavymetal" => Self::Heavymetal,
            "hospitality" => Self::Hospitality,
            "hungerswitch" => Self::Hungerswitch,
            "hustle" => Self::Hustle,
            "iceface" => Self::Iceface,
            "illuminate" => Self::Illuminate,
            "illusion" => Self::Illusion,
            "immunity" => Self::Immunity,
            "imposter" => Self::Imposter,
            "innardsout" => Self::Innardsout,
            "insomnia" => Self::Insomnia,
            "justified" => Self::Justified,
            "keeneye" => Self::Keeneye,
            "klutz" => Self::Klutz,
            "leafguard" => Self::Leafguard,
            "libero" => Self::Libero,
            "lightmetal" => Self::Lightmetal,
            "limber" => Self::Limber,
            "longreach" => Self::Longreach,
            "magicbounce" => Self::Magicbounce,
            "magicguard" => Self::Magicguard,
            "magician" => Self::Magician,
            "magmaarmor" => Self::Magmaarmor,
            "marvelscale" => Self::Marvelscale,
            "megasol" => Self::Megasol,
            "merciless" => Self::Merciless,
            "mimicry" => Self::Mimicry,
            "minus" => Self::Minus,
            "mirrorarmor" => Self::Mirrorarmor,
            "moldbreaker" => Self::Moldbreaker,
            "moody" => Self::Moody,
            "moxie" => Self::Moxie,
            "mummy" => Self::Mummy,
            "noguard" => Self::Noguard,
            "opportunist" => Self::Opportunist,
            "parentalbond" => Self::Parentalbond,
            "pickpocket" => Self::Pickpocket,
            "pickup" => Self::Pickup,
            "piercingdrill" => Self::Piercingdrill,
            "plus" => Self::Plus,
            "poisonheal" => Self::Poisonheal,
            "poisonpoint" => Self::Poisonpoint,
            "pressure" => Self::Pressure,
            "protean" => Self::Protean,
            "punkrock" => Self::Punkrock,
            "purifyingsalt" => Self::Purifyingsalt,
            "queenlymajesty" => Self::Queenlymajesty,
            "quickdraw" => Self::Quickdraw,
            "quickfeet" => Self::Quickfeet,
            "rattled" => Self::Rattled,
            "receiver" => Self::Receiver,
            "ripen" => Self::Ripen,
            "rivalry" => Self::Rivalry,
            "runaway" => Self::Runaway,
            "sandspit" => Self::Sandspit,
            "scrappy" => Self::Scrappy,
            "screencleaner" => Self::Screencleaner,
            "seedsower" => Self::Seedsower,
            "shadowtag" => Self::Shadowtag,
            "arenatrap" => Self::Arenatrap,
            "magnetpull" => Self::Magnetpull,
            "shedskin" => Self::Shedskin,
            "sheerforce" => Self::Sheerforce,
            "shielddust" => Self::Shielddust,
            "simple" => Self::Simple,
            "shieldsdown" => Self::Shieldsdown,
            "skilllink" => Self::Skilllink,
            "sniper" => Self::Sniper,
            "soundproof" => Self::Soundproof,
            "spicyspray" => Self::Spicyspray,
            "stakeout" => Self::Stakeout,
            "stall" => Self::Stall,
            "stalwart" => Self::Stalwart,
            "stancechange" => Self::Stancechange,
            "steadfast" => Self::Steadfast,
            "steelyspirit" => Self::Steelyspirit,
            "stench" => Self::Stench,
            "stickyhold" => Self::Stickyhold,
            "sturdy" => Self::Sturdy,
            "suctioncups" => Self::Suctioncups,
            "superluck" => Self::Superluck,
            "supersweetsyrup" => Self::Supersweetsyrup,
            "supremeoverlord" => Self::Supremeoverlord,
            "surgesurfer" => Self::Surgesurfer,
            "sweetveil" => Self::Sweetveil,
            "symbiosis" => Self::Symbiosis,
            "tangledfeet" => Self::Tangledfeet,
            "telepathy" => Self::Telepathy,
            "thermalexchange" => Self::Thermalexchange,
            "toxicdebris" => Self::Toxicdebris,
            "trace" => Self::Trace,
            "unaware" => Self::Unaware,
            "unburden" => Self::Unburden,
            "unnerve" => Self::Unnerve,
            "unseenfist" => Self::Unseenfist,
            "vitalspirit" => Self::Vitalspirit,
            "wanderingspirit" => Self::Wanderingspirit,
            "waterbubble" => Self::Waterbubble,
            "weakarmor" => Self::Weakarmor,
            "whitesmoke" => Self::Whitesmoke,
            "zerotohero" => Self::Zerotohero,
            _ => Self::Unimplemented,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Item {
    None,
    Unimplemented,
    /// Data-only Mega Stone: the base-form -> Mega mapping plus the refusal to
    /// be taken by Knock Off / Thief / Covet / Trick / Switcheroo.
    MegaStone,
    Leftovers,
    SitrusBerry,
    OranBerry,
    LumBerry,
    LifeOrb,
    ChoiceScarf,
    ChoiceBand,
    ChoiceSpecs,
    FocusSash,
    FocusBand,
    RockyHelmet,
    ExpertBelt,
    BigRoot,
    /// Metronome: consecutive uses of the same move raise its damage
    /// (`conditions:metronome.onModifyDamage`, 4096..8192 over six steps).
    Metronome,
    LightClay,
    DampRock,
    HeatRock,
    SmoothRock,
    IcyRock,
    TerrainExtender,
    // Type-enhancing items (`onBasePowerPriority: 15`, 4915/4096).
    BlackBelt,
    BlackGlasses,
    Charcoal,
    DragonFang,
    FairyFeather,
    HardStone,
    Magnet,
    MetalCoat,
    MiracleSeed,
    MysticWater,
    NeverMeltIce,
    PoisonBarb,
    SharpBeak,
    SilkScarf,
    SilverPowder,
    SoftSand,
    SpellTag,
    TwistedSpoon,
    // Category-enhancing items (`onBasePowerPriority: 16`, 4505/4096).
    MuscleBand,
    WiseGlasses,
    // Defense-side modifiers.
    AssaultVest,
    Eviolite,
    IronBall,
    // Resist berries (`onSourceModifyDamage`).
    BabiriBerry,
    ChartiBerry,
    ChilanBerry,
    ChopleBerry,
    CobBerry,
    ColburBerry,
    HabanBerry,
    KasibBerry,
    KebiaBerry,
    OccaBerry,
    PasshoBerry,
    PayapaBerry,
    RindoBerry,
    RoseliBerry,
    ShucaBerry,
    TangaBerry,
    WacanBerry,
    YacheBerry,
    // Status-curing and PP berries.
    CheriBerry,
    ChestoBerry,
    PechaBerry,
    RawstBerry,
    AspearBerry,
    LeppaBerry,
    PersimBerry,
    // Terrain seeds.
    ElectricSeed,
    GrassySeed,
    MistySeed,
    PsychicSeed,
    // Utility items.
    WhiteHerb,
    MentalHerb,
    AirBalloon,
    WideLens,
    ZoomLens,
    BrightPowder,
    ScopeLens,
    KingsRock,
    ShellBell,
    LightBall,
    Leek,
    BlackSludge,
    StickyBarb,
    NormalGem,
    ShedShell,
    BindingBand,
    QuickClaw,
    RedCard,
    EjectButton,
}
impl Item {
    pub fn compile(id: &str) -> Self {
        match id {
            "" => Self::None,
            "leftovers" => Self::Leftovers,
            "sitrusberry" => Self::SitrusBerry,
            "oranberry" => Self::OranBerry,
            "lumberry" => Self::LumBerry,
            "lifeorb" => Self::LifeOrb,
            "choicescarf" => Self::ChoiceScarf,
            "focussash" => Self::FocusSash,
            "rockyhelmet" => Self::RockyHelmet,
            "expertbelt" => Self::ExpertBelt,
            "bigroot" => Self::BigRoot,
            "metronome" => Self::Metronome,
            "lightclay" => Self::LightClay,
            "damprock" => Self::DampRock,
            "heatrock" => Self::HeatRock,
            "smoothrock" => Self::SmoothRock,
            "icyrock" => Self::IcyRock,
            "terrainextender" => Self::TerrainExtender,
            "choiceband" => Self::ChoiceBand,
            "choicespecs" => Self::ChoiceSpecs,
            "focusband" => Self::FocusBand,
            "blackbelt" => Self::BlackBelt,
            "blackglasses" => Self::BlackGlasses,
            "charcoal" => Self::Charcoal,
            "dragonfang" => Self::DragonFang,
            "fairyfeather" => Self::FairyFeather,
            "hardstone" => Self::HardStone,
            "magnet" => Self::Magnet,
            "metalcoat" => Self::MetalCoat,
            "miracleseed" => Self::MiracleSeed,
            "mysticwater" => Self::MysticWater,
            "nevermeltice" => Self::NeverMeltIce,
            "poisonbarb" => Self::PoisonBarb,
            "sharpbeak" => Self::SharpBeak,
            "silkscarf" => Self::SilkScarf,
            "silverpowder" => Self::SilverPowder,
            "softsand" => Self::SoftSand,
            "spelltag" => Self::SpellTag,
            "twistedspoon" => Self::TwistedSpoon,
            "muscleband" => Self::MuscleBand,
            "wiseglasses" => Self::WiseGlasses,
            "assaultvest" => Self::AssaultVest,
            "eviolite" => Self::Eviolite,
            "ironball" => Self::IronBall,
            "babiriberry" => Self::BabiriBerry,
            "chartiberry" => Self::ChartiBerry,
            "chilanberry" => Self::ChilanBerry,
            "chopleberry" => Self::ChopleBerry,
            "cobaberry" => Self::CobBerry,
            "colburberry" => Self::ColburBerry,
            "habanberry" => Self::HabanBerry,
            "kasibberry" => Self::KasibBerry,
            "kebiaberry" => Self::KebiaBerry,
            "occaberry" => Self::OccaBerry,
            "passhoberry" => Self::PasshoBerry,
            "payapaberry" => Self::PayapaBerry,
            "rindoberry" => Self::RindoBerry,
            "roseliberry" => Self::RoseliBerry,
            "shucaberry" => Self::ShucaBerry,
            "tangaberry" => Self::TangaBerry,
            "wacanberry" => Self::WacanBerry,
            "yacheberry" => Self::YacheBerry,
            "cheriberry" => Self::CheriBerry,
            "chestoberry" => Self::ChestoBerry,
            "pechaberry" => Self::PechaBerry,
            "rawstberry" => Self::RawstBerry,
            "aspearberry" => Self::AspearBerry,
            "leppaberry" => Self::LeppaBerry,
            "persimberry" => Self::PersimBerry,
            "electricseed" => Self::ElectricSeed,
            "grassyseed" => Self::GrassySeed,
            "mistyseed" => Self::MistySeed,
            "psychicseed" => Self::PsychicSeed,
            "whiteherb" => Self::WhiteHerb,
            "mentalherb" => Self::MentalHerb,
            "airballoon" => Self::AirBalloon,
            "widelens" => Self::WideLens,
            "zoomlens" => Self::ZoomLens,
            "brightpowder" => Self::BrightPowder,
            "scopelens" => Self::ScopeLens,
            "kingsrock" => Self::KingsRock,
            "shellbell" => Self::ShellBell,
            "lightball" => Self::LightBall,
            "leek" => Self::Leek,
            "blacksludge" => Self::BlackSludge,
            "stickybarb" => Self::StickyBarb,
            "normalgem" => Self::NormalGem,
            "shedshell" => Self::ShedShell,
            "bindingband" => Self::BindingBand,
            "quickclaw" => Self::QuickClaw,
            "redcard" => Self::RedCard,
            "ejectbutton" => Self::EjectButton,

            _ => Self::Unimplemented,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveBehavior {
    Unimplemented,
    Damage,
    Effect,
    Protect,
    Endure,
    Guard,
    Struggle,
    SideCondition,
    ScreenBreak,
    Weather,
    WeatherBall,
    /// `terrainpulse`: type and power follow the active terrain for a
    /// grounded user.
    TerrainPulse,
    /// `perishsong`: the field-wide three-turn countdown volatile.
    PerishSong,
    /// `haze`: the field-wide boost reset.
    Haze,
    /// `sleeptalk`: only usable while asleep; samples one eligible move from
    /// the user's own moveset and uses it without paying PP.
    SleepTalk,
    /// `substitute`: pays a quarter of the user's maximum HP for a decoy
    /// whose remaining HP lives in the volatile's single value.
    Substitute,
    /// `shedtail`: pays half the user's maximum HP for the same decoy, then
    /// leaves the field; the replacement receives only that decoy.
    ShedTail,
    /// `chillyreception`: an `all`-target status move that sets snow through
    /// the generic weather field, carries the `selfSwitch` pivot and never runs
    /// the move-loop Update pair (the reference routes `all`-target moves
    /// through `tryMoveHit`). Its queued `priorityChargeCallback` action adds
    /// the move's one-turn volatile before any move of the turn.
    ChillyReception,
    /// `gravity`: the five-turn pseudo-weather (grounding, accuracy and the
    /// `flags.gravity` refusal).
    Gravity,
    TrickRoom,
    Terrain,
    /// `trick` / `switcheroo`: item swap with the reference TakeItem refusal
    /// and failed-swap restore semantics.
    Trick,
    /// `fling`: the thrown item sets the action's base power and payload; the
    /// item is consumed by the marker volatile's `onUpdate`.
    Fling,
    /// `topsyturvy`: inverts every nonzero boost stage of the target.
    TopsyTurvy,
    /// `block` / `meanlook`: pin the target with the `trapped` volatile.
    TrapTarget,
    /// `jawlock`: the damaging hit pins both the target and the user.
    JawLock,
    /// `spiritshackle`: a 100% secondary pins the target.
    SpiritShackle,
    /// `aquaring`: a self volatile that heals a sixteenth of the maximum HP
    /// every residual.
    AquaRing,
    /// `spite`: deducts four PP from the target's last move.
    Spite,
    /// `fellstinger`: gains Attack +3 when the hit KOs its target.
    FellStinger,
    /// `lastresort`: fails until every other move slot has been used.
    LastResort,
    /// `counter` / `mirrorcoat`: the user gains a one-turn retaliation
    /// volatile at the start of the turn, records the last qualifying hit in
    /// it, and damages that attacker for `2 * recorded damage`.
    Counter,
    MirrorCoat,
    /// `metalburst` / `comeuppance`: damages the last non-ally attacker for
    /// `floor(1.5 * recorded damage)` and fails unless it hit this turn.
    MetalBurst,
    Comeuppance,
    /// `tidyup`: removes every Substitute on the field, clears the entry
    /// hazards on the user's side and any foe side that has them, then raises
    /// Attack and Speed by one.
    TidyUp,
    /// `courtchange`: swaps the listed side conditions between the two sides
    /// and fails when neither side had one.
    CourtChange,
    /// `wish`: leaves a slot condition that heals the slot's next occupant for
    /// half of the wisher's maximum HP at the following turn's residual.
    Wish,
    /// `healingwish`: faints the user and fully heals the replacement that
    /// enters its slot.
    HealingWish,
    /// `healbell`: cures every party member's status.
    HealBell,
    /// `powertrick` / `powershift`: a self volatile that swaps the user's
    /// stored Attack and Defense for as long as it is up; re-using the move
    /// removes it (swapping back).
    PowerTrick,
    PowerShift,
    /// `powersplit` / `guardsplit`: both sides' stored Attack and Special
    /// Attack (or Defense and Special Defense) become their floored average.
    PowerSplit,
    GuardSplit,
    /// `magneticflux`: raises Defense and Special Defense of every Plus/Minus
    /// holder on the user's side.
    MagneticFlux,
    /// `ingrain`: a self marker that grounds and pins the holder and heals it
    /// at residual order 7.
    Ingrain,
    /// `octolock`: a trapping marker that lowers the holder's defenses every
    /// residual while its source is still active.
    Octolock,
    /// `clearsmog`: damages and then resets the target's boost stages.
    ClearSmog,
    /// `powerswap` / `guardswap`: the two attack (or defense) boost stages swap
    /// with the target through a direct `setBoost` write.
    StatSwap,
    /// `speedswap`: the two stored Speed stats swap, leaving the cached speed
    /// stale until the next `updateSpeed`.
    SpeedSwap,
    /// `helpinghand`: single-turn ally volatile with a stacking BasePower
    /// multiplier.
    HelpingHand,
    /// `followme`: single-turn self volatile that redirects opposing moves.
    FollowMe,
    /// `ragepowder`: Follow Me's powder variant, ignored by a powder-immune
    /// attacker.
    RagePowder,
    /// `allyswitch`: doubles-only position swap between the user and its
    /// partner, gated by the `allyswitch` condition's escalating success
    /// roll on consecutive uses.
    AllySwitch,
    /// `skillswap`: exchange both Pokémon's abilities through the reference
    /// `Battle#skillSwap` helper (End, direct assignment, Start).
    SkillSwap,
    /// `spikes` / `stealthrock` / `toxicspikes` / `stickyweb`: a foe-side
    /// entry hazard with its reference layer cap and restart rules.
    FoeHazard,
    /// `defog`: evasion drop, hazard/screen removal on both sides and
    /// `field.clearTerrain()`.
    Defog,
    /// `corrosivegas`: destroy every adjacent target's held item.
    CorrosiveGas,
    /// `recycle`: restore the user's last consumed item.
    Recycle,
    /// `stockpile`: layered self volatile that raises Defense and Special
    /// Defense by one stage per layer and stores the successful raises so
    /// `onEnd` can reverse them.
    Stockpile,
    /// `swallow`: heals a quarter, half or all of the user's maximum HP from
    /// the stockpile layer count and always consumes the volatile.
    Swallow,
    /// `rest`: fails while already asleep or at full HP (and for the
    /// insomnia family), otherwise forces a three-turn sleep and heals to
    /// full after the status lands.
    Rest,
    /// `destinybond`: drops the `destinybond` volatile (a second consecutive
    /// use instead removes the existing bond and fails); the volatile drags a
    /// foe's move-caused faint down with it and is dropped before the holder's
    /// next non-Destiny-Bond action.
    DestinyBond,
    /// `quash`: forces the target's queued move action behind every other
    /// move action of the turn (`action.order = 201`); fails in singles and
    /// against a target with no queued move.
    Quash,
    /// `revivalblessing`: fails with no fainted party member, otherwise puts
    /// the `revivalblessing` slot condition on the user and asks the side to
    /// pick a fainted party member to revive at half HP.
    RevivalBlessing,
    /// `bellydrum`: pays half the user's maximum HP to set Attack to +6 (fails
    /// at half HP or less, at a capped Attack, and for a one-HP maximum).
    BellyDrum,
    /// `acupressure`: raises one random stat that is not already +6 by two
    /// stages (fails when every stage is capped).
    Acupressure,
    /// `synthesis` / `moonlight` / `morningsun`: heals half the user's maximum
    /// HP, two thirds in sun and a quarter in any other weather.
    WeatherHeal,
    /// `upperhand`: fails outright unless the target still has a queued move
    /// action whose *declaration* priority is positive and whose category is
    /// not Status.
    UpperHand,
    /// `round`: a queued Round action anywhere in the turn's queue jumps to
    /// the front of the queue and resolves immediately after this one with
    /// its base power doubled.
    Round,
    /// `healpulse`: heals the target for ceil(baseMaxhp / 2); a full-HP
    /// target is refused with `NOT_FAIL`.
    HealPulse,
    /// `painsplit`: both active HP values become the floor of their average
    /// (at least one), bypassing heal gates.
    PainSplit,
    /// `endeavor`: fixed damage equal to the target's HP minus the user's,
    /// refused outright unless the user is strictly lower.
    Endeavor,
    /// `bugbite` / `pluck`: eat the target's held Berry through the reference
    /// `takeItem` + `Eat` sequence, applying its `onEat` effect to the user.
    ItemSteal,
}

/// Cold payload of a ported two-turn move. Every field is transcribed from
/// the pinned reference declaration for that move id; the battle path reads
/// the struct by numeric id and never re-derives it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChargeSpec {
    /// Weather ids that complete the charge immediately (`solarbeam` /
    /// `solarblade`: sun; `electroshot`: rain).
    pub instant_weather: Vec<Id>,
    /// The move reads `attacker.effectiveWeather(undefined, true)` (the
    /// `message` argument), which only differs from the plain read under Mega
    /// Sol. Mega Sol itself stays an explicit operational error, so this flag
    /// exists to keep the recipe exact rather than to approximate it.
    pub instant_weather_message: bool,
    /// Self boost applied during the prepare step (`electroshot` / `meteorbeam`
    /// +1 Sp. Atk, `skullbash` +1 Defense).
    pub prepare_boost: [i8; 7],
    /// The charge volatile grants semi-invulnerability (`onInvulnerability`).
    pub semi_invulnerable: bool,
    /// Move ids whose `onInvulnerability` handler returns undefined, letting
    /// the move hit a semi-invulnerable target (Earthquake vs. Dig…).
    pub invuln_exceptions: Vec<Id>,
    /// `onSourceModifyDamage` doubles damage from these move ids.
    pub damage_double: Vec<Id>,
    /// `onSourceBasePower` doubles base power from these move ids.
    pub power_double: Vec<Id>,
    /// `onImmunity`: underground targets ignore sandstorm and hail damage.
   pub weather_immune: bool,
    /// The move's own volatile duration (fly/dig/dive/bounce and the Phantom
    /// Force family declare `duration: 2`; Solar Beam and friends declare no
    /// condition, so their marker volatile has no duration).
    pub volatile_duration: Option<u16>,
    /// `onBasePower`: Solar Beam / Solar Blade are halved while any weak
    /// weather (rain, sandstorm, snow/hail) is in effect.
    pub half_in_weak_weather: bool,
}

/// Reference `Protect`-family contact punishment, executed by the volatile
/// that actually blocked the hit. Each variant is an exact transcription of
/// the pinned move condition's `onTryHit` contact branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectPunish {
    None,
    /// Spiky Shield: `this.damage(source.baseMaxhp / 8, source, target)`.
    DamageEighthMaxHp,
    /// Baneful Bunker: `source.trySetStatus('psn', target, ...)`.
    Poison,
    /// King's Shield: `this.boost({atk: -1}, source, target, ...)`.
    AttackDown,
}

impl MoveBehavior {
    pub fn compile(id: &str) -> Self {
        match id {
            "psychic" | "energyball" | "dragonclaw" | "dragonpulse" | "seedbomb"
            | "smartstrike" | "highhorsepower" | "megahorn" | "xscissor" | "slash"
            | "nightslash" | "shadowclaw" | "aerialace" | "aquajet" | "aquatail" | "hydropump"
            | "surf" | "earthquake" | "hypervoice" | "dazzlinggleam" | "powergem" | "tackle"
            | "pound" | "scratch" | "quickattack" | "vinewhip" | "watergun" | "gust"
            | "wingattack" | "peck" | "drillpeck" | "psychocut" | "razorleaf" | "mudslap"
            | "snore" | "facade" | "burningjealousy" => {
                Self::Damage
            }
            "flamethrower" | "icebeam" | "thunderbolt" | "shadowball" | "darkpulse"
            | "bodyslam" | "scald" | "heatwave" | "rockslide" | "discharge" | "zapcannon"
            | "icywind" | "snarl" | "ancientpower" | "closecombat" | "overheat" | "leafstorm"
            | "dracometeor" | "bulletpunch" | "firepunch" | "icepunch" | "thunderpunch"
            | "crunch" | "bite" | "airslash" | "ironhead" | "playrough" => Self::Damage,
            "doubleedge" | "takedown" | "submission" | "wildcharge" | "flareblitz"
            | "woodhammer" | "bravebird" | "headsmash" | "volttackle" | "absorb" | "megadrain"
            | "gigadrain" | "drainpunch" | "drainingkiss" | "hornleech" | "leechlife"
            | "oblivionwing" | "paraboliccharge" => Self::Damage,
            "toxic" | "willowisp" | "thunderwave" | "poisonpowder" | "sleeppowder" | "spore"
            | "hypnosis" | "swordsdance" | "calmmind" | "agility" | "irondefense" | "nastyplot"
            | "charm" | "faketears" | "growl" | "recover" | "slackoff" | "sandattack"
            | "doubleteam" | "sweetscent" => Self::Effect,
            "protect" => Self::Protect,
            // The Protect family shares the `stall` gate and the priority-3
            // blocking phase; each move carries its own blocking volatile.
            "detect" | "spikyshield" | "banefulbunker" | "kingsshield" => Self::Protect,
            // Endure rides the same stall gate but clamps damage instead of
            // blocking the hit.
            "endure" => Self::Endure,
            // Duration-one side conditions that block spread/priority moves.
            "wideguard" | "quickguard" => Self::Guard,
            "struggle" => Self::Struggle,
            "tailwind" | "reflect" | "lightscreen" | "auroraveil" | "safeguard" => {
                Self::SideCondition
            }
            "brickbreak" | "psychicfangs" => Self::ScreenBreak,
            "raindance" | "sunnyday" | "sandstorm" | "snowscape" => Self::Weather,
            "weatherball" => Self::WeatherBall,
            "terrainpulse" => Self::TerrainPulse,
            "perishsong" => Self::PerishSong,
            "haze" => Self::Haze,
            "substitute" => Self::Substitute,
            "shedtail" => Self::ShedTail,
            "chillyreception" => Self::ChillyReception,
            "gravity" => Self::Gravity,
            "sleeptalk" => Self::SleepTalk,
            "trickroom" => Self::TrickRoom,
            "electricterrain" | "grassyterrain" | "mistyterrain" | "psychicterrain" => {
                Self::Terrain
            }
            "trick" | "switcheroo" => Self::Trick,
            "fling" => Self::Fling,
            "powerswap" | "guardswap" => Self::StatSwap,
            "speedswap" => Self::SpeedSwap,
            "topsyturvy" => Self::TopsyTurvy,
            "clearsmog" => Self::ClearSmog,
            "block" | "meanlook" => Self::TrapTarget,
            "jawlock" => Self::JawLock,
            "spiritshackle" => Self::SpiritShackle,
            "aquaring" => Self::AquaRing,
            "spite" => Self::Spite,
            "fellstinger" => Self::FellStinger,
            "lastresort" => Self::LastResort,
            "counter" => Self::Counter,
            "mirrorcoat" => Self::MirrorCoat,
            "metalburst" => Self::MetalBurst,
            "comeuppance" => Self::Comeuppance,
            "tidyup" => Self::TidyUp,
            "courtchange" => Self::CourtChange,
            "wish" => Self::Wish,
            "healingwish" => Self::HealingWish,
            "healbell" => Self::HealBell,
            "powertrick" => Self::PowerTrick,
            "powershift" => Self::PowerShift,
            "powersplit" => Self::PowerSplit,
            "guardsplit" => Self::GuardSplit,
            "magneticflux" => Self::MagneticFlux,
            "ingrain" => Self::Ingrain,
            "octolock" => Self::Octolock,
            "helpinghand" => Self::HelpingHand,
            "followme" => Self::FollowMe,
            "ragepowder" => Self::RagePowder,
            "allyswitch" => Self::AllySwitch,
            "skillswap" => Self::SkillSwap,
            "spikes" | "stealthrock" | "toxicspikes" | "stickyweb" => Self::FoeHazard,
            "defog" => Self::Defog,
            "corrosivegas" => Self::CorrosiveGas,
            "recycle" => Self::Recycle,
            "stockpile" => Self::Stockpile,
            "swallow" => Self::Swallow,
            "rest" => Self::Rest,
            "destinybond" => Self::DestinyBond,
            "quash" => Self::Quash,
            "revivalblessing" => Self::RevivalBlessing,
            "bellydrum" => Self::BellyDrum,
            "acupressure" => Self::Acupressure,
            "synthesis" | "moonlight" | "morningsun" => Self::WeatherHeal,
            "upperhand" => Self::UpperHand,
            "round" => Self::Round,
            "healpulse" => Self::HealPulse,
            "painsplit" => Self::PainSplit,
            "endeavor" => Self::Endeavor,
            "bugbite" | "pluck" => Self::ItemSteal,
            _ => Self::Unimplemented,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HitEffect {
    pub boosts: [i8; 7],
    pub status: Id,
    pub volatile: Id,
    pub heal: Option<[u16; 2]>,
}

/// Ported reference `basePowerCallback` formulas. The cold loader resolves a
/// move's callback key to one of these; the battle path evaluates it with
/// local state only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasePowerKind {
    Acrobatics,
    ElectroBall,
    Eruption,
    Flail,
    GrassKnot,
    GyroBall,
    HardPress,
    HeatCrash,
    Hex,
    InfernalParade,
    LastRespects,
    LowKick,
    /// `ragefist`: `min(350, 50 + 50 * timesAttacked)`.
    RageFist,
    /// `tripleaxel`: `20 * move.hit` (the current hit number).
    TripleAxel,
    /// `stompingtantrum`: doubles when the user's previous move failed.
    StompingTantrum,
    PowerTrip,
    RisingVoltage,
    /// `beatup`: `5 + floor(setSpecies.baseStats.atk / 10)` of the ally the
    /// current hit consumes from the move's captured party list.
    BeatUp,
    /// `spitup`: 100 power per stored stockpile layer.
    Stockpile,
    /// `payback` / `avalanche`: the declared power doubles when the target has
    /// already acted this turn and did not just switch in.
    Payback,
    Avalanche,
    /// `watershuriken`: the declared power; the reference's Greninja-Ash /
    /// Battle Bond branch is unreachable because Battle Bond stays an explicit
    /// operational error.
    WaterShuriken,
    /// `assurance`: doubles when the target already took damage this turn.
    HurtTarget,
    /// `temperflare`: doubles when the user's previous move failed.
    TemperFlare,
}

impl BasePowerKind {
    pub fn compile(key: &str) -> Option<Self> {
        Some(match key {
            "moves:acrobatics.basePowerCallback" => Self::Acrobatics,
            "moves:electroball.basePowerCallback" => Self::ElectroBall,
            "moves:eruption.basePowerCallback" | "moves:waterspout.basePowerCallback" => {
                Self::Eruption
            }
            "moves:flail.basePowerCallback" | "moves:reversal.basePowerCallback" => Self::Flail,
            "moves:grassknot.basePowerCallback" => Self::GrassKnot,
            "moves:gyroball.basePowerCallback" => Self::GyroBall,
            "moves:hardpress.basePowerCallback" => Self::HardPress,
            "moves:heatcrash.basePowerCallback" | "moves:heavyslam.basePowerCallback" => {
                Self::HeatCrash
            }
            "moves:hex.basePowerCallback" => Self::Hex,
            "moves:infernalparade.basePowerCallback" => Self::InfernalParade,
            "moves:lastrespects.basePowerCallback" => Self::LastRespects,
            "moves:lowkick.basePowerCallback" => Self::LowKick,
            "moves:ragefist.basePowerCallback" => Self::RageFist,
            "moves:tripleaxel.basePowerCallback" => Self::TripleAxel,
            "moves:stompingtantrum.basePowerCallback" => Self::StompingTantrum,
            "moves:payback.basePowerCallback" => Self::Payback,
            "moves:avalanche.basePowerCallback" => Self::Avalanche,
            "moves:watershuriken.basePowerCallback" => Self::WaterShuriken,
            "moves:powertrip.basePowerCallback" | "moves:storedpower.basePowerCallback" => {
                Self::PowerTrip
            }
            "moves:risingvoltage.basePowerCallback" => Self::RisingVoltage,
            "moves:beatup.basePowerCallback" => Self::BeatUp,
            "moves:spitup.basePowerCallback" => Self::Stockpile,
            "moves:assurance.basePowerCallback" => Self::HurtTarget,
            "moves:temperflare.basePowerCallback" => Self::TemperFlare,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryEffect {
    pub chance: u8,
    pub target: HitEffect,
    pub own: Option<HitEffect>,
}

/// `items:<id>.fling`: how the Fling move resolves the thrown item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlingKind {
    /// Damage only (`leftovers`, `ironball`, every Mega Stone).
    Plain,
    /// `isBerry`: the target eats the thrown Berry through its `onEat`.
    Berry,
    /// `fling.status`: a guaranteed status on the target.
    Status(Id),
    /// `fling.volatileStatus`: a guaranteed volatile on the target.
    Volatile(Id),
    /// The Mental Herb callback: clears the target's attract/taunt/encore/
    /// torment/disable/healblock volatiles.
    MentalHerb,
    /// The White Herb callback: clears the target's negative boosts.
    WhiteHerb,
    /// A `fling.effect` with no native port: Fling stays an explicit error for
    /// that item instead of guessing.
    Unsupported,
}

/// One row of the pinned `fling` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlingSpec {
    pub base_power: u16,
    pub kind: FlingKind,
}

#[derive(Debug, Clone)]
pub struct NativeEffects {
    pub abilities: Vec<Ability>,
    /// `items:<id>.fling`: the Fling move's per-item table. `None` means the
    /// item declares no `fling` data, so Fling fails outright with it.
    pub fling_items: Vec<Option<FlingSpec>>,
    /// `conditions:fling`: the marker volatile whose `onUpdate` consumes the
    /// thrown item after the action.
    pub fling: Id,
    /// `conditions:trapped`: the pseudo-type marker Block, Mean Look, Jaw Lock
    /// and Spirit Shackle add (Ghost types stay immune to the actual trap).
    pub trapped: Id,
    /// `conditions:trapper`: the silent partner marker the linked
    /// `addVolatile('trapped', ..., 'trapper')` adds to the trapper.
    pub trapper: Id,
    /// `moves:aquaring.condition`: the residual heal volatile.
    pub aqua_ring: Id,
    /// `moves:ingrain.condition`: the grounding/fixing self marker.
    pub ingrain: Id,
    /// `moves:octolock.condition`: the trapping marker whose residual lowers
    /// the holder's Defense and Special Defense.
    pub octolock: Id,
    /// `moves:powertrick.condition` / `moves:powershift.condition`: the
    /// Attack/Defense swap volatiles (identical conditions under two ids).
    pub power_trick: Id,
    pub power_shift: Id,
    /// `moves:wish.condition` / `moves:healingwish.condition`: slot conditions
    /// that resolve on the next residual / on the replacement's switch-in.
    pub wish: Id,
    pub healing_wish: Id,
    /// The three moves themselves (their `damage`/`onHit` payloads read the
    /// conditions above).
    pub wish_move: Id,
    pub healing_wish_move: Id,
    pub heal_bell_move: Id,
    /// `moves:counter.condition`: the one-turn retaliation volatile that
    /// records the last qualifying Physical hit (`moves:mirrorcoat.condition`
    /// does the same for Special hits).
    pub counter: Id,
    pub mirrorcoat: Id,
    /// `moves:counter` / `moves:mirrorcoat` (the moves, not the conditions):
    /// their `damageCallback` reads the matching volatile.
    pub counter_move: Id,
    pub mirrorcoat_move: Id,
    /// `moves:powerswap`: the Stat Swap behaviour needs to know which pair of
    /// stages it moves.
    pub power_swap_move: Id,
    /// Pinned `flags.breakable` abilities: only these can be ignored by the
    /// active move (Mold Breaker / Teravolt / Turboblaze / a move-level
    /// `ignoreAbility`). Indexed by ability id.
    pub breakable_abilities: Vec<bool>,
    /// `flags.cantsuppress`: abilities that Mummy / Wandering Spirit / Skill
    /// Swap may not overwrite or exchange.
    pub no_suppress_abilities: Vec<bool>,
    /// `flags.failskillswap`: abilities that refuse the Skill Swap exchange
    /// on either side.
    pub no_skill_swap_abilities: Vec<bool>,
    /// `flags.noentrain` / `flags.failroleplay`: the ability-transfer moves
    /// refuse these holders.
    pub no_entrain_abilities: Vec<bool>,
    pub fail_role_play_abilities: Vec<bool>,
    /// The `simple` and `truant` ability ids (Simple Beam's gates).
    pub simple_ability: Id,
    pub truant_ability: Id,
    pub moves: Vec<MoveBehavior>,
    pub items: Vec<Item>,
    /// Pinned `flags.notrace` abilities: Trace never copies them (Trace itself
    /// carries the flag, so a Trace holder cannot copy another Trace).
    pub no_trace_abilities: Vec<bool>,
    pub choice_lock: Id,
    /// Pinned `onDisableMove` condition declarations (id -> reference
    /// `resolvePriority` sub-order). `endTurn`'s `runEvent('DisableMove')`
    /// handler list is speed-sorted, so membership and sub-order are
    /// RNG-visible; the loader fails closed if the set changes.
    pub disable_move_conditions: std::collections::BTreeMap<Id, i32>,
    /// Pinned `onFoeDisableMove` condition declarations (Imprison), collected
    /// from each live active foe of the event target.
    pub foe_disable_move_conditions: std::collections::BTreeMap<Id, i32>,
    /// Pinned `onDisableMove` ability declarations (Gorilla Tactics).
    pub disable_move_abilities: std::collections::BTreeMap<Id, i32>,
    /// Pinned `onDisableMove` item declarations (Assault Vest).
    pub disable_move_items: std::collections::BTreeMap<Id, i32>,
    /// Exact Champions base-form -> Mega-form mappings, indexed by held item.
    pub mega_stones: Vec<Vec<(Id, Id)>>,
    /// `items[*].isBerry`: the reference's `takeItem`-then-`Eat` steal moves
    /// (Bug Bite, Pluck) only act on berries.
    pub berry_items: Vec<bool>,
    pub protect: Id,
    pub stall: Id,
    /// The `revivalblessing` slot condition (Revival Blessing's revive
    /// protocol marker).
    pub revival_blessing: Id,
    /// `focusenergy` / `dragoncheer`: the two crit-ratio volatiles.
    pub focus_energy: Id,
    pub dragon_cheer: Id,
    /// The `charge` volatile (move Charge / Electromorphosis) and the move id
    /// it is consumed by.
    pub charge: Id,
    pub charge_move: Id,
    /// `noretreat`: the self-trap marker volatile.
    pub no_retreat: Id,
    /// `safeguard`: the side condition that refuses foe-sourced statuses.
    pub safeguard: Id,
    /// `smackdown`: the marker that grounds an airborne target.
    pub smack_down: Id,
    /// `conditions:commanded`: a commanded user (Tatsugiri's Commander) never
    /// sets a `selfSwitch` pivot flag.
    pub commanded: Id,
    /// `moves:batonpass`: the `selfSwitch: 'copyvolatile'` pivot whose
    /// replacement copies the outgoing Pokémon's boosts and volatiles.
    pub baton_pass_move: Id,
    /// `moves:shedtail`: the `selfSwitch: 'shedtail'` pivot whose replacement
    /// receives only the outgoing Pokémon's decoy.
    pub shed_tail_move: Id,
    /// `moves:gravity.condition`: the five-turn pseudo-weather that grounds
    /// every active Pokémon, raises accuracy by 6840/4096 and refuses
    /// `flags.gravity` moves.
    pub gravity: Id,
    /// The two aerial charge moves whose markers Gravity removes on start.
    pub fly_move: Id,
    pub bounce_move: Id,
    /// `moves:curse.condition`: the Ghost curse's quarter-max-HP drain.
    pub curse: Id,
    /// `moves:chillyreception.condition`: the one-turn volatile added by the
    /// queued move's `priorityChargeCallback` action.
    pub chilly_reception: Id,
    /// Conditions declaring `noCopy`: `copyVolatileFrom` refuses to transfer
    /// them (Disable, Encore, Yawn, Smack Down, Stockpile, ...).
    pub no_copy_conditions: Vec<Id>,
    /// Conditions declaring an `onCopy` callback (Gastro Acid, Power Shift,
    /// Power Trick). None of them is reachable from a ported effect, so a
    /// transfer stays an explicit operational error rather than a silent
    /// no-op.
    pub copy_callback_conditions: Vec<Id>,
    /// Additional Protect-family volatiles that block hits in the same
    /// `hitStepTryHitEvent` phase (priority 3).
    pub spiky_shield: Id,
    pub baneful_bunker: Id,
    pub kings_shield: Id,
    /// `endure` volatile: clamps incoming move damage to `hp - 1`.
    pub endure: Id,
    /// Duration-one side conditions that block spread / priority moves.
    pub wide_guard: Id,
    pub quick_guard: Id,
    /// `moves:toxicspikes.condition`: a layer-based entry hazard side
    /// condition (no duration) that poisons or badly poisons grounded
    /// switch-ins and is absorbed by Poison types.
    pub toxic_spikes: Id,
    /// `spikes` / `stealthrock` / `stickyweb` entry-hazard condition ids.
    pub spikes: Id,
    pub stealth_rock: Id,
    pub sticky_web: Id,
    /// `thief` / `covet`: the two `onAfterHit` item-stealing moves.
    pub thief_move: Id,
    pub covet_move: Id,
    /// Snow-only screen that halves both damage categories.
    pub aurora_veil: Id,
    /// Single-turn redirection / support volatiles.
    pub helping_hand: Id,
    pub follow_me: Id,
    pub rage_powder: Id,
    /// `moves:allyswitch.condition`: the position-swap volatile whose stored
    /// value is the escalating consecutive-use success counter (3 -> 9 -> ...
    /// -> 729).
    pub ally_switch: Id,
    /// `moves:stockpile.condition`: the layered Defense/Special Defense
    /// volatile consumed by Spit Up and Swallow.
    pub stockpile: Id,
    /// The four self-destructing moves refused by `abilities:damp`.
    pub damp_moves: [Id; 4],
    /// Volatile that skips the holder's next action.
    pub must_recharge: Id,
    /// `twoturnmove` condition: locks the holder into the charging move and
    /// carries the stored target location for the second turn.
    pub two_turn_move: Id,
    /// `lockedmove` condition: the Outrage / Thrash / Petal Dance / Raging
    /// Fury rampage lock. Stores the locked move and its rolled true duration.
    pub locked_move: Id,
    /// `moves:yawn.condition`: two-turn countdown that ends in sleep.
    pub yawn: Id,
    /// `moves:yawn` (the move, not the volatile condition): `onTryHit` refuses
    /// a target that already has a status or cannot fall asleep.
    pub yawn_move: Id,
    /// `moves:firstimpression`: the id disabled by its own `onDisableMove`.
    pub first_impression: Id,
    /// `moves:toxic` (the move, not the `tox` status): a Poison-type attacker
    /// hits through semi-invulnerability.
    pub toxic_move: Id,
    /// `moves:helpinghand` (the move, not the volatile condition): the
    /// invulnerability step short-circuits for it.
    pub helping_hand_move: Id,
    /// `moves:round`: the Round chain's queued-action priority effect and the
    /// `move.sourceEffect` marker that doubles a chained Round's base power.
    pub round: Id,
    /// `moves:destinybond` volatile condition: the bond's `onFaint` trigger
    /// and the `onBeforeMove` / `onMoveAborted` drop points.
    pub destiny_bond: Id,
    /// `moves:destinybond` move id: the `onBeforeMove` drop is skipped when
    /// the holder's attempted move is Destiny Bond itself.
    pub destiny_bond_move: Id,
    /// `moves:ceaselessedge.onAfterHit|onAfterSubDamage`: scatters one Spikes
    /// layer onto the foe side when the hit lands or a decoy absorbs it.
    pub ceaseless_edge: Id,
    /// `moves:stoneaxe.onAfterHit|onAfterSubDamage`: sets Stealth Rock on the
    /// foe side when the hit lands or a decoy absorbs it.
    pub stone_axe: Id,
    /// `moves:roost.condition`: one-turn Flying removal for the caster.
    pub roost: Id,
    /// `moves:glaiverush.condition`: the user's next move always lands, and
    /// incoming damage is doubled until the volatile is consumed.
    pub glaive_rush: Id,
    /// `moves:minimize.condition`: the evasion volatile whose
    /// `onSourceModifyDamage` doubles the damage of `flags.minimize` moves and
    /// whose `onAccuracy` makes those moves bypass the accuracy roll.
    pub minimize: Id,
    /// `partiallytrapped` condition: binding moves' damage/trap volatile.
    pub partially_trapped: Id,
    /// `perishsong` condition: four-tick countdown that ends in a faint.
    pub perish_song: Id,
    /// `leechseed` condition: drains the holder into the seeding slot.
    pub leech_seed: Id,
    /// `moves:substitute.condition`: the user's damage-absorbing decoy. The
    /// volatile's single value is the decoy's remaining HP.
    pub substitute: Id,
    /// `conditions:metronome` volatile: the held-item consecutive-use counter.
    pub metronome: Id,
    /// Held item `abilityshield`: the holder's ability cannot be ignored by the
    /// active move. The item declares no other behaviour yet, so it stays out of
    /// the ported `Item` table; this id only drives the suppression gate.
    pub ability_shield: Id,
    /// Volatile that disables and refuses sound moves for two turns.
    pub throat_chop: Id,
    /// `healblock` condition: refuses every HP recovery, disables the holder's
    /// `heal`-flag moves and refuses one already committed to them. Applied by
    /// Psychic Noise for two turns (the past-generation Heal Block move would
    /// use five, and Persistent seven, but neither is legal in M-C).
    pub heal_block: Id,
    /// Disguise's species family, resolved once at load: the undisguised forms
    /// absorb the first damaging move, the busted forms do not.
    pub mimikyu: Id,
    pub mimikyu_totem: Id,
    pub mimikyu_busted: Id,
    pub mimikyu_busted_totem: Id,
    /// Stance Change's two Aegislash formes and the King's Shield move id.
    /// `aegislash` is the base forme the submitted set always starts in.
    pub aegislash: Id,
    pub aegislash_blade: Id,
    /// Paldea Tauros forms whose Raging Bull takes its own primary type.
    pub tauros_paldea_combat: Id,
    pub tauros_paldea_blaze: Id,
    pub tauros_paldea_aqua: Id,
    pub kings_shield_move: Id,
    /// `move:encore` volatile: locks the holder into its last move.
    pub encore: Id,
    /// `move:taunt` volatile: refuses Status moves.
    pub taunt: Id,
    /// `move:disable` volatile: refuses one recorded move. Also applied by the
    /// Cursed Body ability.
    pub disable: Id,
    /// `move:imprison` volatile: hides moves shared with the user from foes.
    pub imprison: Id,
    /// `move:torment` volatile: refuses the holder's last move.
    pub torment: Id,
    /// `Mefirst` move id, exempt from Taunt's disable pass.
    pub me_first: Id,
    /// Mental Herb item id, the Champions Encore queue-change gate.
    pub mental_herb: Id,
    /// Volatile with a 2..5 turn timer and a 33% self-hit chance.
    pub confusion: Id,
    /// Ability id for the Trick/Switcheroo `onTryImmunity` refusal.
    pub sticky_hold: Id,
    pub flash_fire: Id,
    /// Ability-granted volatile that doubles Speed while the holder has no item.
    pub unburden: Id,
    pub struggle: Id,
    pub ground: Id,
    pub fire: Id,
    pub normal: Id,
    pub fairy: Id,
    pub water: Id,
    pub grass: Id,
    pub bug: Id,
    pub ice: Id,
    pub electric: Id,
    pub poison_type: Id,
    pub steel: Id,
    pub burn: Id,
    pub paralysis: Id,
    pub sleep: Id,
    pub freeze: Id,
    pub poison: Id,
    pub toxic: Id,
    pub flinch: Id,
    pub drain: Id,
    pub recoil: Id,
    pub tailwind: Id,
    pub reflect: Id,
    pub light_screen: Id,
    pub rain: Id,
    pub sun: Id,
    pub sand: Id,
    pub snow: Id,
    pub rock: Id,
    pub dark: Id,
    pub ghost: Id,
    pub fighting: Id,
    pub trick_room: Id,
    pub electric_terrain: Id,
    pub grassy_terrain: Id,
    pub misty_terrain: Id,
    pub psychic_terrain: Id,
    pub flying: Id,
    pub psychic: Id,
    pub dragon: Id,
    pub quake_moves: [Id; 3],
    /// Ported action-local callbacks per move id (see `hook`).
    pub move_hooks: Vec<u64>,
    /// Fake Out move id, used by the ported Champions `onDisableMove`.
    pub fake_out: Id,
}

impl NativeEffects {
    /// Every volatile whose `onTryHit` blocks a normal hit. Order is the
    /// reference creation order used to pick the punishment of the volatile
    /// that actually blocked; the list stays tiny and local to the target.
    pub fn protection_volatiles(&self) -> [Id; 4] {
        [
            self.protect,
            self.spiky_shield,
            self.baneful_bunker,
            self.kings_shield,
        ]
    }

    pub fn protect_punish(&self, volatile: Id) -> ProtectPunish {
        if volatile == self.spiky_shield {
            ProtectPunish::DamageEighthMaxHp
        } else if volatile == self.baneful_bunker {
            ProtectPunish::Poison
        } else if volatile == self.kings_shield {
            ProtectPunish::AttackDown
        } else {
            ProtectPunish::None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub side: u8,
    pub roster: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueuedKind {
    BeforeTurn,
    Move,
    /// Reference `priorityChargeMove` (queue order 107): the queued move's
    /// `priorityChargeCallback` runs before every move of the turn.
    PriorityCharge,
    /// Reference `beforeTurnMove` (queue order 5): the queued move's
    /// `beforeTurnCallback` runs before every move of the turn (Counter and
    /// Mirror Coat add their one-turn retaliation volatile).
    BeforeTurnMove,
    Switch,
    RunSwitch,
    Mega,
    Residual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedAction {
    pub kind: QueuedKind,
    pub actor: Option<Entity>,
    pub move_slot: u8,
    pub move_id: Id,
    /// `BattleQueue#prioritizeAction` writes the causing effect onto the
    /// action; the Round chain is the only move that reads it (`Round`'s
    /// base power doubles when `move.sourceEffect` is `round`).
    pub source_effect: Id,
    pub target_location: i8,
    pub destination: u8,
    pub priority: crate::queue::Priority,
}
