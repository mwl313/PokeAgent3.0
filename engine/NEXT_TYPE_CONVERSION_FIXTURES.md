# Type conversion fixture scout

Read-only next-family plan, 2026-10-07. These are validator-approved setups and focused oracle probes, **not implemented complete differential fixtures**. The current 192-fixture generator/corpus was not changed by this scout. Full legal M-C and called-effect closure remains the engine objective.

## Reference and demonstrated behavior

Pinned official Showdown commit `14546894d86f9589ac11130c510bbe73b6968665`, format `gen9championsvgc2026regmc`.

Source callbacks: `data/abilities.ts` Aerilate 57, Galvanize 1598, Normalize 2987, Pixilate 3297, Refrigerate 3813. No Champions override exists. Move preparation is `sim/battle-actions.ts:useMoveInner`, including `ModifyType` before `ModifyMove`, target resolution, redirection, hit checks and BasePower.

Aerilate/Pixilate/Refrigerate/Galvanize change eligible Normal moves to Flying/Fairy/Ice/Electric with ModifyType priority -1. Normalize has priority +1 and changes eligible moves of any type, including already-Normal moves, to Normal. All use BasePower priority 23 and the action-local `typeChangerBoosted === this.effect` marker. The exact coefficient is **4915/4096**, not an approximate floating-point 1.2 or an old-generation 1.3. With no competing modifier the reference result is `floor((power * 4915 + 2047) / 4096)`; preserve the common chained-modifier path when another hook contributes.

The callback makes no activation/reveal log entry. Do not force an ability reveal just because a hidden calculation used Sylveon's or Aurorus's ability. Mega form/ability observations follow the existing Mega contract.

## Legal holders and full-six witnesses

Validator-derived starting holders: Pixilate Sylveon; Refrigerate Aurorus. Allowed Mega forms: Pixilate Mega Altaria and Mega Gardevoir; Aerilate Mega Pinsir and Mega Salamence; Refrigerate Mega Glalie. Galvanize and Normalize have **no legal starting holder or permitted form** in this pin. Do not create an illegal Alolan Golem or Delcatty battle.

Every candidate below was checked through `ReferenceSession`, which runs `TeamValidator.validateTeam` on both full six-member teams. Every set uses level 50, Serious nature, Stat Points `{hp:32, def:17, spd:17}` and no item unless listed. Candidate is P1 roster 0. P1 rosters 1–5 are:

1. Goodra-Hisui / Shell Armor / Dragon Pulse, Protect.
2. Torterra / Shell Armor / Seed Bomb, Protect.
3. Falinks / Battle Armor / Smart Strike, Protect.
4. Samurott / Shell Armor / Aqua Jet, Protect.
5. Hydreigon / Levitate / Dragon Pulse, Protect.

P2 full six is Goodra-Hisui, Torterra, Falinks, Perrserker, Samurott, Hydreigon, using the same abilities/moves above with Perrserker / Battle Armor / Iron Head, Protect. Preview selects rosters `[0,1,4,5]` on both sides. Oracle seed `[1,2,3,2000]` was used for focused turn-one probes. Candidate executes move 1, with Mega when holding its stone; partner Dragon Pulse; opposing active members use their first damaging move. These probes establish actual callbacks, not eventual battle completion.

| Candidate set | Item | Actual first action / result |
| --- | --- | --- |
| Sylveon / Pixilate / Hyper Voice, Quick Attack, Weather Ball, Protect | none | Hyper Voice Normal → Fairy; BasePower 90 → 108 for both targets |
| Aurorus / Refrigerate / Hyper Voice, Weather Ball, Rock Slide, Protect | none | Hyper Voice Normal → Ice; BasePower 90 → 108 for both targets |
| Altaria / Cloud Nine / Hyper Voice, Double-Edge, Weather Ball, Protect | Altarianite | Mega acquires Pixilate before action; Hyper Voice Normal → Fairy, 90 → 108 |
| Gardevoir / Synchronize / Hyper Voice, Psychic, Double Team, Protect | Gardevoirite | Mega acquires Pixilate before action; Hyper Voice Normal → Fairy, 90 → 108 |
| Pinsir / Hyper Cutter / Quick Attack, X-Scissor, Protect | Pinsirite | Mega acquires Aerilate; Quick Attack Normal → Flying, 40 → 48 |
| Salamence / Intimidate / Hyper Voice, Double-Edge, Dragon Claw, Protect | Salamencite | Mega acquires Aerilate; Hyper Voice Normal → Flying, 90 → 108 |
| Glalie / Ice Body / Body Slam, Ice Beam, Weather Ball, Protect | Glalitite | Mega acquires Refrigerate; Body Slam Normal → Ice, 85 → 102 |

Rejected alternatives matter: Pinsir cannot learn Double-Edge, Glalie cannot learn Hyper Voice, and Gardevoir cannot learn Weather Ball in this format. Use Salamence for Aerilate physical plus special coverage and Aurorus for Refrigerate special coverage.

## Decisive complete-fixture drafts

Preserve all existing inputs. Derive new complete battles from the legal witnesses, validate the final full teams, and require natural completion rather than synthetic draw/truncation.

- **Starting Pixilate/Refrigerate:** Sylveon Hyper Voice and Quick Attack; Aurorus Hyper Voice and Rock Slide as an eligible/ineligible control. Assert actual converted type, marker identity, fixed-point BasePower and resulting damage/STAB/weakness. Converted spread moves retain their original spread targeting. Swap in Sylveon Double-Edge for a physical recoil control; validate that changed set separately.
- **Mega acquisition:** delay each Mega until turn two to witness an unconverted first action and converted second action. Also keep at least one turn-one Mega/action case. Test permanent conversion after switch-out/reentry; no retained boost marker on the next unrelated action. Altaria loses Cloud Nine on Mega, so stored weather must become effective before that turn's Weather Ball and later weather effects.
- **Physical/special and priority:** Salamence Double-Edge/Hyper Voice, Pinsir Quick Attack, Glalie Body Slam, Aurorus Hyper Voice. Quick Attack retains its priority. Use the already supported Psychic Terrain branch for a grounded defending target; conversion cannot bypass enemy priority protection. Use actual recoil/secondary effects, not a plain-damage replacement for Double-Edge or Body Slam.
- **Defensive interactions:** converted Ice attacks versus an actual Thick Fat Mega Venusaur; physical converted Ice versus an Ice target in snow to test the defense modifier; converted Fairy versus a Dragon target, and converted Flying versus a Rock target. Existing effect ordering and integer damage comparisons must decide outcomes.
- **Status conversion:** Gardevoir Double Team is legal and Normal → Fairy after Mega, while the self accuracy boost still works and no BasePower callback is invented for status moves. Protect offers self-target controls on all witnesses. Altaria Sing and Sylveon Yawn are legal targeted Normal status choices but are currently unsupported dependencies; do not describe them as already covered. An actual converted status callback must be probed if one is added.
- **No-op handler ordering:** non-Normal Psychic/Dragon Claw/Rock Slide and excluded Weather Ball must still execute the registered BasePower priority-23 callback and leave the relay unchanged. Assert marker absent and exact RNG/state parity. Avoid inventing a speed tie between two conversion holders: onBasePower is local to the attacking holder, so merely placing two equal-speed holders on the field does not create two participating conversion callbacks. If another real eligible handler shares the event, collect actual handler identities/priorities/speeds before claiming tied-handler evidence.

## Weather Ball exclusions and absorption controls

The following separate full-six probes used seed `[1,2,3,2010]`, Sylveon / Pixilate / Weather Ball, Hyper Voice, Protect at P1 roster 0, and the same four backline fillers above. P1 roster 1 was Goodra-Hisui / Shell Armor / Dragon Pulse, Protect, except the Cloud Nine case uses Altaria / Cloud Nine with those moves. Enemy rosters 0–1 and their results were:

| Control | Enemy active sets | Actual Weather Ball result |
| --- | --- | --- |
| Clear | Goodra-Hisui / Shell Armor / Dragon Pulse, Protect; Perrserker / Battle Armor / Iron Head, Protect | Normal, BP 50, no conversion marker; BasePower relay 50 |
| Rain | Heliolisk / Dry Skin / Thunderbolt, Protect; Pelipper / Drizzle / Surf, Protect | Water, BP 100, no marker; Dry Skin TryHit returns null |
| Sun | Arcanine / Flash Fire / Flamethrower, Protect; Ninetales / Drought / Flamethrower, Protect | Fire, BP 100, no marker; Flash Fire TryHit returns null |
| Stored rain suppressed | Rain enemy above plus allied Cloud Nine Altaria | Normal, BP 50, no marker; BasePower relay 50 |

P2 backline is Torterra, Falinks, Samurott, Hydreigon as above. Preview `[0,1,4,5]`, Sylveon Weather Ball at enemy roster 0; partner and both opponents Protect. All four full-team validations and real callback probes succeeded. These are good complete-fixture seeds: require actual absorbed Water/Fire effects plus an ordinary target that reaches BasePower to prove the lack of the 20% bonus, then switch Cloud Nine away or Mega it to restore weather.

All ilate exclusions are Judgment, Multi-Attack, Natural Gift, Revelation Dance, Techno Blast, Terrain Pulse, Weather Ball. Normalize additionally excludes Hidden Power and Struggle. Weather Ball is the only listed exclusion learned by these legal starting witnesses (Sylveon, Aurorus, Altaria, Glalie). Do not add illegal exclusion moves to a complete battle. Cover the remaining callback branches with honest isolated primitive tests and later legal called-move/acquisition evidence. Max/Z/Tera exceptions are in reference callbacks but those battle resources are not legal in this M-C format. Struggle's own preparation uses the typeless `???` type; test it without pretending that its Dex Normal type guarantees an ilate conversion.

## Galvanize / Normalize frontier

There is currently no validated legal battle source for either ability. Primitive tests should nevertheless exercise the shared handlers: Galvanize-converted single-target Normal → Electric must enter Lightning Rod redirection before hit checks, boost Volt Absorb/Motor Drive targets, and spread conversion must skip redirection; Normalize-converted Fire → Normal must bypass Flash Fire, and Electric → Normal must bypass Lightning Rod. Normalize already-Normal moves still receive its marker/boost. Source/target ordering and self-target exclusions remain the absorption family's contracts.

Skill Swap, Role Play, Entrainment, Trace or Transform do not by themselves supply a missing ability. A claimed legal acquisition witness needs a complete permitted source chain. Current conservative full-Dex closure is uncertified; neither absence of a starting holder nor a synthetic illegal fixture certifies that frontier.

## Native dependencies and verification boundary

As inspected during this scout, conversion abilities/action-local conversion marker are not implemented. Existing support covers Hyper Voice, Quick Attack, Double-Edge, Weather Ball, Psychic, Dragon Claw, Rock Slide, Double Team, Protect, the listed weather/absorption/redirect families and the relevant ordinary damage/recoil paths. New full behaviors are needed for **Synchronize**, **Hyper Cutter**, and **Body Slam's 30% paralysis** before all seven witnesses can run natively. Hyper Beam/Giga Impact require recharge and are not a plain-move shortcut. Round has combination/pull-queue behavior; Snore has sleep gating/flinch; Sing and Yawn have their own status lifecycle dependencies. Do not register these as incomplete plain attacks to make a fixture pass.

After implementation, require every new complete battle to pass exact boundary HP/stats/status/PP/RNG, traces, snapshot/replay and reused batch equivalence. Actual execution guards must prove conversion and its marker, exact BP and no-op, original targeting/priority, Mega transition, Weather Ball exclusion/absorption, and any handler tie that truly occurs. This note certifies only the legal witnesses and focused reference observations described above.
