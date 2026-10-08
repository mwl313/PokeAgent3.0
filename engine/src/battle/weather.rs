//! Weather is field state, independent of the caster remaining active/alive.
use super::*;

impl BattleState {
    pub(super) fn weather_id(&self, dex: &Dex) -> Id {
        [
            dex.effects.rain,
            dex.effects.sun,
            dex.effects.sand,
            dex.effects.snow,
        ]
        .into_iter()
        .find(|id| self.field.contains_key(id))
        .unwrap_or(0)
    }

    pub(super) fn weather_suppressed(&self, dex: &Dex) -> bool {
        // Zero HP queued for fainting still suppresses until the End callback.
        // Suppression-changing abilities/volatiles remain guarded unsupported.
        self.active_entities(false).into_iter().any(|e| {
            let p = self.mon(e);
            !p.ability_ending
                && matches!(
                    dex.effects.abilities[p.ability as usize],
                    Ability::CloudNine | Ability::AirLock
                )
        })
    }

    pub(super) fn effective_weather(&self, dex: &Dex) -> Id {
        let weather = self.weather_id(dex);
        if weather == 0 || self.weather_suppressed(dex) {
            0
        } else {
            weather
        }
    }

    pub(super) fn ability_end(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::FlashFire
            && self.mon(e).hp > 0
            && self
                .mon_mut(e)
                .volatiles
                .remove(&dex.effects.flash_fire)
                .is_some()
        {
            self.emit(
                EventKind::EffectEnd,
                e,
                None,
                EffectRef::Condition(dex.effects.flash_fire),
                0,
                false,
            )?;
        }
        if matches!(
            dex.effects.abilities[self.mon(e).ability as usize],
            Ability::CloudNine | Ability::AirLock
        ) {
            self.mon_mut(e).ability_ending = true;
            // Departing/fainting holder is still included in WeatherChange sort.
            self.field_change_order();
        }
        // `abilities:unburden.onEnd` removes the volatile. Its condition has no
        // End callback, so the removal emits nothing.
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Unburden {
            self.mon_mut(e).volatiles.remove(&dex.effects.unburden);
        }
        // `abilities:supremeoverlord.onEnd`: the frozen boost ends with the
        // ability (the reference's `-end fallenN` marker is silent).
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Supremeoverlord {
            self.mon_mut(e).supreme_overlord_fallen = 0;
        }
        Ok(())
    }

    pub(super) fn field_change_order(&mut self) -> SmallVec<[Entity; 4]> {
        let mut entities = self.active_entities(false);
        let speeds: SmallVec<[(Entity, i32); 4]> = entities
            .iter()
            .map(|&e| (e, self.mon(e).cached_speed))
            .collect();
        speed_sort(&mut entities, &mut self.rng, |e| Priority {
            speed: speeds.iter().find(|(x, _)| x == e).unwrap().1,
            ..Default::default()
        });
        entities
    }

    pub(super) fn start_weather(
        &mut self,
        dex: &Dex,
        source: Entity,
        id: Id,
        ability: bool,
    ) -> Result<bool> {
        let old = self.weather_id(dex);
        if old == id {
            return Ok(false);
        }
        if ![
            dex.effects.rain,
            dex.effects.sun,
            dex.effects.sand,
            dex.effects.snow,
        ]
        .contains(&id)
        {
            return Err(EngineError::Unsupported(format!("weather {id}")));
        }
        let rock = if id == dex.effects.rain {
            Item::DampRock
        } else if id == dex.effects.sun {
            Item::HeatRock
        } else if id == dex.effects.sand {
            Item::SmoothRock
        } else {
            Item::IcyRock
        };
        let duration = if dex.effects.items[self.mon(source).item as usize] == rock {
            8
        } else {
            5
        };
        if old != 0 {
            self.field.remove(&old);
            self.emit(
                EventKind::FieldEffectEnd,
                source,
                None,
                EffectRef::Condition(old),
                0,
                false,
            )?;
        }
        self.field.insert(
            id,
            EffectState {
                id,
                duration: Some(duration),
                source: Some((
                    if source.side == 0 {
                        SideId::P1
                    } else {
                        SideId::P2
                    },
                    source.roster,
                )),
                ..Default::default()
            },
        );
        if ability {
            self.reveal_ability(source)?;
        }
        self.emit(
            EventKind::FieldEffectStart,
            source,
            None,
            EffectRef::Condition(id),
            i32::from(duration),
            false,
        )?;
        // EachEvent WeatherChange sorts active Pokémon even with no callbacks,
        // then runs each active's own onWeatherChange handlers in that order.
        let order = self.field_change_order();
        self.weather_change_handlers(dex, &order)?;
        Ok(true)
    }

    pub(super) fn weather_upkeep(&mut self, dex: &Dex, id: Id) -> Result<()> {
        let Some(state) = self.field.get_mut(&id) else {
            return Ok(());
        };
        let source = state
            .source
            .map(|(s, roster)| Entity {
                side: s.index() as u8,
                roster,
            })
            .unwrap();
        let duration = state.duration.as_mut().unwrap();
        *duration -= 1;
        let expired = *duration == 0;
        for knowledge in &mut self.knowledge {
            if let Some(effect) = knowledge.field.get_mut(&id)
                && effect.duration.known
            {
                effect.duration.value -= 1;
            }
        }
        if expired {
            self.field.remove(&id);
            self.emit(
                EventKind::FieldEffectEnd,
                source,
                None,
                EffectRef::Condition(id),
                0,
                false,
            )?;
            // Reference `Field#clearWeather`: the ending weather runs the same
            // WeatherChange set as a fresh start.
            let order = self.field_change_order();
            self.weather_change_handlers(dex, &order)?;
            return Ok(());
        }
        let suppressed = self.weather_suppressed(dex);
        // Rain/sun still run EachEvent Weather (and Update) when suppressed;
        // sand/snow guard the entire call with field.isWeather.
        if suppressed && (id == dex.effects.sand || id == dex.effects.snow) {
            return Ok(());
        }
        for target in self.field_change_order() {
            if suppressed {
                continue;
            }
            let p = self.mon(target);
            let immune = [dex.effects.rock, dex.effects.ground, dex.effects.steel]
                .iter()
                .any(|t| p.types.contains(t))
                // `moves:dig|dive.condition.onImmunity`: a charging user
                // ignores sandstorm and hail chip damage.
                || self
                    .charging_spec(dex, target)
                    .is_some_and(|spec| spec.weather_immune)
                || matches!(
                    dex.effects.abilities[p.ability as usize],
                    Ability::SandRush | Ability::SandForce | Ability::SandVeil | Ability::Overcoat
                );
            let ability = dex.effects.abilities[p.ability as usize];
            if id == dex.effects.sand && !immune {
                self.indirect_damage(
                    dex,
                    target,
                    source,
                    u32::from((p.stats[0] / 16).max(1)),
                    EffectRef::Condition(id),
                )?;
            }
            // These ability Weather handlers share the already sorted target
            // order. Their supported weather has no field onWeather handler,
            // so no additional handler tie draw is required.
            if (ability == Ability::RainDish && id == dex.effects.rain)
                || (ability == Ability::IceBody && id == dex.effects.snow)
                || (ability == Ability::DrySkin && id == dex.effects.rain)
            {
                let p = self.mon(target);
                // Rain Dish / Ice Body / Dry Skin heal through `this.heal`, so
                // Heal Block refuses the recovery (the ability stays hidden
                // because only the heal message would name it).
                if p.hp > 0 && p.hp < p.stats[0] && !self.heal_blocked(dex, target) {
                    let denominator = if ability == Ability::DrySkin { 8 } else { 16 };
                    let amount = (p.stats[0] / denominator).max(1).min(p.stats[0] - p.hp);
                    self.mon_mut(target).hp += amount;
                    self.reveal_ability(target)?;
                    self.emit(
                        EventKind::Heal,
                        target,
                        None,
                        EffectRef::Ability(self.mon(target).ability),
                        i32::from(amount),
                        true,
                    )?;
                }
            } else if matches!(ability, Ability::SolarPower | Ability::DrySkin)
                && id == dex.effects.sun
                && self.mon(target).hp > 0
            {
                self.reveal_ability(target)?;
                self.indirect_damage(
                    dex,
                    target,
                    target,
                    u32::from((self.mon(target).stats[0] / 8).max(1)),
                    EffectRef::Ability(self.mon(target).ability),
                )?;
            }
        }
        // Gen 7+ EachEvent Weather invokes Update before the residual handler's
        // faint-message boundary, allowing berries to activate after sand damage.
        self.each_update(dex)?;
        self.process_faints(dex, true)
    }

    /// Reference `Field#setWeather` / `Field#clearWeather` close with
    /// `eachEvent('WeatherChange', sourceEffect)`: every active runs its own
    /// `onWeatherChange` handlers in the already-sorted field-change order. A
    /// fainted slot keeps its place in the sort but collects no handlers.
    fn weather_change_handlers(&mut self, dex: &Dex, order: &[Entity]) -> Result<()> {
        for &e in order {
            if self.mon(e).fainted {
                continue;
            }
            self.forecast_weather_change(dex, e)?;
        }
        Ok(())
    }

    /// `abilities:forecast.onWeatherChange` (and its `onStart`, which issues the
    /// same singleEvent): Castform's forme follows the effective weather. The
    /// handler runs with the ability as the active effect, so
    /// `pokemon.effectiveWeather()` takes no Mega Sol override - it is the
    /// plain field weather, or the empty string while Cloud Nine/Air Lock
    /// suppresses it.
    pub(super) fn forecast_weather_change(&mut self, dex: &Dex, e: Entity) -> Result<()> {
        if dex.effects.abilities[self.mon(e).ability as usize] != Ability::Forecast {
            return Ok(());
        }
        let castform = dex.id("species", "Castform")?;
        let mon = self.mon(e);
        if mon.transformed || dex.species[mon.base_species as usize].base_species != castform {
            return Ok(());
        }
        let weather = self.effective_weather(dex);
        let target = if weather == dex.effects.sun {
            dex.id("species", "Castform-Sunny")?
        } else if weather == dex.effects.rain {
            dex.id("species", "Castform-Rainy")?
        } else if weather == dex.effects.snow {
            dex.id("species", "Castform-Snowy")?
        } else {
            castform
        };
        if self.mon(e).species == target {
            return Ok(());
        }
        self.forme_change(dex, e, target)
    }

    /// `Pokemon#effectiveWeather` for the pinned regulation: a Mega Sol holder
    /// resolves under sun no matter what the field weather is. The reference
    /// only applies the override when the source effect is that ability, a move
    /// or a weather, so ability-owned handlers (Chlorophyll, Solar Power,
    /// Leaf Guard, Hydration, Sand Veil, ...) deliberately keep using
    /// `effective_weather`.
    pub(super) fn mon_weather(&self, dex: &Dex, e: Entity) -> Id {
        if dex.effects.abilities[self.mon(e).ability as usize] == Ability::Megasol {
            dex.effects.sun
        } else {
            self.effective_weather(dex)
        }
    }

    /// `conditions:sunnyday|raindance.onWeatherModifyDamage` gate on the
    /// defender's `effectiveWeather`, but `Pokemon#effectiveWeather` keys its
    /// Mega Sol override on `battle.activePokemon` - the move's user - and on
    /// the effect currently running, so under a Mega Sol attacker *both* the
    /// attacker's and the defender's query resolve to sun. `abilities:megasol`
    /// relays the sun handler inside a `priorityEvent`, whose fast exit keeps
    /// the field weather's own handler from running alongside it, so the rule
    /// is exactly the move user's `effectiveWeather` view.
    pub(super) fn weather_damage_modifier(
        &self,
        dex: &Dex,
        actor: Entity,
        _defender: Entity,
        move_type: Id,
    ) -> u32 {
        // The handler's `defender.effectiveWeather() != <own id>` gate cannot
        // reject under the modeled item set: both sides read the same
        // `mon_weather` view (a Utility Umbrella holder would flip it, and is
        // still an explicit gap).
        let rule = self.mon_weather(dex, actor);
        if (rule == dex.effects.rain && move_type == dex.effects.water)
            || (rule == dex.effects.sun && move_type == dex.effects.fire)
        {
            6144
        } else if (rule == dex.effects.rain && move_type == dex.effects.fire)
            || (rule == dex.effects.sun && move_type == dex.effects.water)
        {
            2048
        } else {
            4096
        }
    }
}
