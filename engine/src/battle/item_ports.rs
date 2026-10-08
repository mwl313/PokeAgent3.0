//! Native ports of held-item behaviour beyond the initial item set. Every
//! predicate and modifier below is transcribed from the pinned reference
//! (`data/items.ts`); nothing here approximates an unported callback.
use super::*;
use super::hooks::HookList;

/// Fixed type for each type-enhancing item (`onBasePowerPriority: 15`).
fn type_item_type(dex: &Dex, item: Item) -> Option<Id> {
    Some(match item {
        Item::BlackBelt => dex.effects.fighting,
        Item::BlackGlasses => dex.effects.dark,
        Item::Charcoal => dex.effects.fire,
        Item::DragonFang => dex.effects.dragon,
        Item::FairyFeather => dex.effects.fairy,
        Item::HardStone => dex.effects.rock,
        Item::Magnet => dex.effects.electric,
        Item::MetalCoat => dex.effects.steel,
        Item::MiracleSeed => dex.effects.grass,
        Item::MysticWater => dex.effects.water,
        Item::NeverMeltIce => dex.effects.ice,
        Item::PoisonBarb => dex.effects.poison_type,
        Item::SharpBeak => dex.effects.flying,
        Item::SilkScarf => dex.effects.normal,
        Item::SilverPowder => dex.effects.bug,
        Item::SoftSand => dex.effects.ground,
        Item::SpellTag => dex.effects.ghost,
        Item::TwistedSpoon => dex.effects.psychic,
        _ => return None,
    })
}

/// Resist berries. `false` means the berry requires a super-effective hit;
/// Chilan is the Normal-type exception that also halves neutral hits.
fn resist_berry_type(dex: &Dex, item: Item) -> Option<(Id, bool)> {
    Some(match item {
        Item::ChilanBerry => (dex.effects.normal, false),
        Item::BabiriBerry => (dex.effects.steel, true),
        Item::ChartiBerry => (dex.effects.rock, true),
        Item::ChopleBerry => (dex.effects.fighting, true),
        Item::CobBerry => (dex.effects.bug, true),
        Item::ColburBerry => (dex.effects.dark, true),
        Item::HabanBerry => (dex.effects.dragon, true),
        Item::KasibBerry => (dex.effects.ghost, true),
        Item::KebiaBerry => (dex.effects.poison_type, true),
        Item::OccaBerry => (dex.effects.fire, true),
        Item::PasshoBerry => (dex.effects.water, true),
        Item::PayapaBerry => (dex.effects.psychic, true),
        Item::RindoBerry => (dex.effects.grass, true),
        Item::RoseliBerry => (dex.effects.fairy, true),
        Item::ShucaBerry => (dex.effects.ground, true),
        Item::TangaBerry => (dex.effects.bug, true),
        Item::WacanBerry => (dex.effects.electric, true),
        Item::YacheBerry => (dex.effects.ice, true),
        _ => return None,
    })
}

fn push(
    list: &mut HookList,
    state: &BattleState,
    holder: Entity,
    priority: i32,
    modifier: u32,
) {
    list.push((
        Priority {
            priority: priority * 10000,
            speed: state.mon(holder).cached_speed,
            sub_order: 8,
            ..Default::default()
        },
        modifier,
    ));
}

/// Adds item-owned handlers to the shared event list with the reference
/// priority and the holder's cached speed, preserving exact tie ordering.
pub(super) fn collect_hooks(
    state: &mut BattleState,
    dex: &Dex,
    event: ModifierEvent,
    context: MoveContext<'_>,
    hooks: &mut HookList,
) -> Result<()> {
    let MoveContext {
        actor,
        target,
        move_data: m,
        effectiveness,
        ..
    } = context;
    let attacking_item = dex.effects.items[state.mon(actor).item as usize];
    let defending_item = dex.effects.items[state.mon(target).item as usize];
    match event {
        ModifierEvent::BasePower => {
            if type_item_type(dex, attacking_item) == Some(m.move_type) {
                push(hooks, state, actor, 15, 4915);
            }
            match attacking_item {
                Item::MuscleBand if m.category == Category::Physical => {
                    push(hooks, state, actor, 16, 4505);
                }
                Item::WiseGlasses if m.category == Category::Special => {
                    push(hooks, state, actor, 16, 4505);
                }
                _ => {}
            }
        }
        ModifierEvent::Attack => {
            if attacking_item == Item::ChoiceBand {
                push(hooks, state, actor, 1, 6144);
            }
            if attacking_item == Item::LightBall && is_base_species(dex, actor, state, "pikachu") {
                push(hooks, state, actor, 1, 8192);
            }
        }
        ModifierEvent::SpecialAttack => {
            if attacking_item == Item::ChoiceSpecs {
                push(hooks, state, actor, 1, 6144);
            }
            if attacking_item == Item::LightBall && is_base_species(dex, actor, state, "pikachu") {
                push(hooks, state, actor, 1, 8192);
            }
        }
        ModifierEvent::Defense => {
            if defending_item == Item::Eviolite && is_nfe(state, dex, target) {
                push(hooks, state, target, 2, 6144);
            }
        }
        ModifierEvent::SpecialDefense => {
            // Sand's Rock SpD boost is applied first with the same sequential
            // rounding the reference uses (priority 10 before 2 and 1).
            if defending_item == Item::Eviolite && is_nfe(state, dex, target) {
                push(hooks, state, target, 2, 6144);
            }
            if defending_item == Item::AssaultVest {
                push(hooks, state, target, 1, 6144);
            }
        }
        ModifierEvent::Damage => {
            if let Some((berry_type, needs_super_effective)) = resist_berry_type(dex, defending_item)
            {
                let matches = m.move_type == berry_type
                    && (!needs_super_effective || effectiveness > 0);
                if matches && state.mon(target).hp > 0 && !state.unnerve_blocks_eat(dex, target) {
                    // The reference eats the berry inside the handler itself;
                    // a refused Unnerve eat leaves the berry and the damage
                    // unmodified.
                    state.consume_item(dex, target)?;
                    push(hooks, state, target, 0, 2048);
                    // `abilities:ripen.onSourceModifyDamage` (priority -1): the
                    // recorded weaken-berry eat adds a second halving.
                    if dex.effects.abilities[state.mon(target).ability as usize] == Ability::Ripen {
                        push(hooks, state, target, -1, 2048);
                    }
                }
            }
        }
    }
    Ok(())
}

fn is_base_species(dex: &Dex, _e: Entity, state: &BattleState, name: &str) -> bool {
    let species = state.mon(_e).species as usize;
    let base = dex.species[species].base_species as usize;
    dex.names["species"][base] == name
}

fn is_nfe(state: &BattleState, dex: &Dex, e: Entity) -> bool {
    dex.species[state.mon(e).species as usize].nfe
}

/// `onModifyCritRatio` items: Scope Lens adds one stage; Leek adds two for
/// Farfetch'd and Sirfetch'd.
pub(super) fn crit_ratio_bonus(state: &BattleState, dex: &Dex, holder: Entity) -> u8 {
    let item = dex.effects.items[state.mon(holder).item as usize];
    let base = dex.species[state.mon(holder).species as usize].base_species as usize;
    match item {
        Item::ScopeLens => 1,
        Item::Leek => {
            let name = &dex.names["species"][base];
            if name == "farfetchd" || name == "sirfetchd" {
                2
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// `onUpdate` item effects: status berries, Leppa Berry, Black Sludge and
/// Sticky Barb. Runs inside `item_update` for every Update event.
pub(super) fn update(state: &mut BattleState, dex: &Dex, e: Entity) -> Result<()> {
    let p = state.mon(e);
    if p.hp == 0 {
        return Ok(());
    }
    match dex.effects.items[p.item as usize] {
        Item::CheriBerry if p.status == dex.effects.paralysis => {
            if !state.unnerve_blocks_eat(dex, e) {
                state.consume_item(dex, e)?;
                state.cure_status(e)?;
            }
        }
        Item::ChestoBerry if p.status == dex.effects.sleep => {
            if !state.unnerve_blocks_eat(dex, e) {
                state.consume_item(dex, e)?;
                state.cure_status(e)?;
            }
        }
        Item::PechaBerry if p.status == dex.effects.poison || p.status == dex.effects.toxic => {
            if !state.unnerve_blocks_eat(dex, e) {
                state.consume_item(dex, e)?;
                state.cure_status(e)?;
            }
        }
        Item::RawstBerry if p.status == dex.effects.burn => {
            if !state.unnerve_blocks_eat(dex, e) {
                state.consume_item(dex, e)?;
                state.cure_status(e)?;
            }
        }
        Item::AspearBerry if p.status == dex.effects.freeze => {
            if !state.unnerve_blocks_eat(dex, e) {
                state.consume_item(dex, e)?;
                state.cure_status(e)?;
            }
        }
        Item::LeppaBerry => {
            let slot = state
                .mon(e)
                .moves
                .iter()
                .position(|slot| slot.pp == 0)
                .or_else(|| {
                    state
                        .mon(e)
                        .moves
                        .iter()
                        .position(|slot| slot.pp < dex.moves[slot.id as usize].pp)
                });
            if let Some(slot) = slot
                && !state.unnerve_blocks_eat(dex, e)
            {
                let move_id = state.mon(e).moves[slot].id as usize;
                let max = dex.moves[move_id].pp;
                state.consume_item(dex, e)?;
                state.mon_mut(e).moves[slot].pp =
                    state.mon(e).moves[slot].pp.saturating_add(10).min(max);
                state.mon_mut(e).base_moves[slot].pp =
                    state.mon(e).base_moves[slot].pp.saturating_add(10).min(max);
            }
        }
        _ => return Ok(()),
    }
    Ok(())
}

/// Residual item handlers (`onResidualOrder` 5/28/29 in the pinned data).
pub(super) fn residual_item(state: &mut BattleState, dex: &Dex, e: Entity, item: Id) -> Result<()> {
    if state.mon(e).hp == 0 || state.mon(e).item != item {
        return Ok(());
    }
    let item_enum = dex.effects.items[item as usize];
    match item_enum {
        Item::BlackSludge => {
            let amount = state.mon(e).stats[0] / 16;
            if state.mon(e).types.contains(&dex.effects.poison_type) {
                state.item_heal(dex, e, amount, false)?;
            } else {
                state.indirect_damage(dex, e, e, u32::from(amount), EffectRef::Item(item))?;
            }
        }
        Item::StickyBarb => {
            let amount = u32::from(state.mon(e).stats[0] / 8);
            state.indirect_damage(dex, e, e, amount, EffectRef::Item(item))?;
        }
        Item::WhiteHerb => white_herb(state, dex, e)?,
        _ => {}
    }
    Ok(())
}

/// Item-owned switch-in behaviour: Choice-item lock clearing, terrain seeds,
/// White Herb, and the Air Balloon start message.
pub(super) fn start(state: &mut BattleState, dex: &Dex, e: Entity) -> Result<()> {
    if state.mon(e).hp == 0 {
        return Ok(());
    }
    let item = dex.effects.items[state.mon(e).item as usize];
    if matches!(item, Item::ChoiceBand | Item::ChoiceSpecs)
        && state
            .mon(e)
            .volatiles
            .contains_key(&dex.effects.choice_lock)
    {
        state.mon_mut(e).volatiles.remove(&dex.effects.choice_lock);
    }
    apply_seed(state, dex, e, item)?;
    if item == Item::WhiteHerb {
        white_herb(state, dex, e)?;
    }
    // `items:metronome.onStart`: the item adds its counter volatile. The
    // volatile has no `onRestart`, so a re-add while it already exists is a
    // silent no-op (reference `Pokemon#addVolatile`).
    if item == Item::Metronome && !state.mon(e).volatiles.contains_key(&dex.effects.metronome) {
        let order = state.allocate_effect_order()?;
        state.mon_mut(e).volatiles.insert(
            dex.effects.metronome,
            EffectState {
                id: dex.effects.metronome,
                values: vec![0, 0],
                effect_order: order,
                effect_order_assigned: true,
                ..Default::default()
            },
        );
    }
    Ok(())
}

/// `onTerrainChange` seeds: consumed when the matching terrain becomes active.
pub(super) fn terrain_change(state: &mut BattleState, dex: &Dex, e: Entity) -> Result<()> {
    if state.mon(e).hp == 0 {
        return Ok(());
    }
    let item = dex.effects.items[state.mon(e).item as usize];
    apply_seed(state, dex, e, item)
}

/// Reference `eachEvent('TerrainChange')`: all seed holders resolve in the
/// ordinary speed order (with the exact tie shuffling) before any of them
/// consumes its item. Mimicry's `onTerrainChange` is a handler of the same
/// event and joins the identical sort, so a Mimicry holder ties (and shuffles)
/// against a seed holder at the same speed.
pub(super) fn terrain_change_event(state: &mut BattleState, dex: &Dex) -> Result<()> {
    let mut handlers: SmallVec<[(bool, Entity, Priority); 6]> = state
        .active_entities(false)
        .into_iter()
        .filter(|e| {
            matches!(
                dex.effects.items[state.mon(*e).item as usize],
                Item::ElectricSeed | Item::GrassySeed | Item::MistySeed | Item::PsychicSeed
            )
        })
        .map(|e| {
            (
                false,
                e,
                Priority {
                    speed: state.mon(e).cached_speed,
                    ..Default::default()
                },
            )
        })
        .collect();
    handlers.extend(
        state
            .active_entities(false)
            .into_iter()
            .filter(|e| {
                dex.effects.abilities[state.mon(*e).ability as usize] == Ability::Mimicry
            })
            .map(|e| {
                (
                    true,
                    e,
                    Priority {
                        speed: state.mon(e).cached_speed,
                        ..Default::default()
                    },
                )
            }),
    );
    speed_sort(&mut handlers, &mut state.rng, |h| h.2);
    for (mimicry, e, _) in handlers {
        if mimicry {
            state.mimicry_terrain_change(dex, e)?;
        } else {
            terrain_change(state, dex, e)?;
        }
    }
    Ok(())
}

/// `onAnySwitchIn`, `onAnyAfterMega` and `onAnyAfterMove` White Herb checks.
pub(super) fn white_herb_event(state: &mut BattleState, dex: &Dex) -> Result<()> {
    for e in state.active_entities(false) {
        white_herb(state, dex, e)?;
    }
    Ok(())
}

fn apply_seed(state: &mut BattleState, dex: &Dex, e: Entity, item: Item) -> Result<()> {
    let terrain = state.terrain_id(dex);
    let (matches, stat) = match item {
        Item::ElectricSeed => (terrain == dex.effects.electric_terrain, 1),
        Item::GrassySeed => (terrain == dex.effects.grassy_terrain, 2),
        Item::MistySeed => (terrain == dex.effects.misty_terrain, 3),
        Item::PsychicSeed => (terrain == dex.effects.psychic_terrain, 4),
        _ => return Ok(()),
    };
    if !matches {
        return Ok(());
    }
    state.consume_item(dex, e)?;
    let mut boosts = [0i8; 7];
    boosts[stat] = 1;
    state.boost(dex, e, e, boosts, BoostCause::Item)?;
    Ok(())
}

/// `whiteherb`: clears the holder's negative boosts and consumes the item.
/// Runs on switch-in, after Mega, after any move, and each residual.
pub(super) fn white_herb(state: &mut BattleState, dex: &Dex, e: Entity) -> Result<()> {
    if state.mon(e).hp == 0
        || dex.effects.items[state.mon(e).item as usize] != Item::WhiteHerb
        || !state.mon(e).boosts.iter().any(|b| *b < 0)
    {
        return Ok(());
    }
    state.consume_item(dex, e)?;
    for boost in state.mon_mut(e).boosts.iter_mut() {
        if *boost < 0 {
            *boost = 0;
        }
    }
    Ok(())
}
