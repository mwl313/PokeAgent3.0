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
        // EachEvent WeatherChange sorts active Pokémon even with no callbacks.
        self.field_change_order();
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
            self.field_change_order();
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
                if p.hp > 0 && p.hp < p.stats[0] {
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

    pub(super) fn weather_damage_modifier(&self, dex: &Dex, move_type: Id) -> u32 {
        let weather = self.effective_weather(dex);
        if (weather == dex.effects.rain && move_type == dex.effects.water)
            || (weather == dex.effects.sun && move_type == dex.effects.fire)
        {
            6144
        } else if (weather == dex.effects.rain && move_type == dex.effects.fire)
            || (weather == dex.effects.sun && move_type == dex.effects.water)
        {
            2048
        } else {
            4096
        }
    }
}
