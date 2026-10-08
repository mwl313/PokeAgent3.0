// Development-only corpus for `abilities:forecast` (Castform):
// - the forme follows the weather the moment it changes (Drizzle on the
//   opposing lead, or a Sunny Day used mid-battle), swapping the base type,
// - the same WeatherChange set runs when the weather ends, so Castform reverts,
// - without weather the forme never moves.
// Every fixture is a complete legal synthetic reference battle with the pinned
// Showdown state recorded at every decision boundary.
import {createScaffold} from './fixture_scaffold.mjs';
const {ids, setOf, offensive, foeTeam, logHas, everHas, runTrials} = createScaffold();

const team = (...heads) => [
  ...heads,
  ...foeTeam().filter(p => !heads.some(head => head.species === p.species)),
].slice(0, 6);
const castform = () => setOf('Castform', 'Forecast', ['Protect', 'Weather Ball']);
const politoed = () => setOf('Politoed', 'Drizzle', ['Protect', 'Surf', 'Ice Beam']);
const sunny = () => offensive('Torterra', 'Shell Armor', ['Sunny Day', 'Protect']);
const bulky = () => setOf('Snorlax', 'Thick Fat', ['Protect', 'Body Slam']);

const TRIALS = [
  {
    name: 'forecast_follows_the_rain_on_switch_in',
    p1: () => team(castform(), bulky()),
    p2: () => team(politoed()),
    script: [{p1: ['protect', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'forecast'},
    verify(fixture, session) {
      if (!logHas(session, /\|-formechange\|p1a: s0\|Castform-Rainy\|\[msg\]\|\[from\] ability: Forecast/)) {
        return 'the rainy forme was never announced';
      }
      if (!everHas(fixture, 0, 0, p => p.species === ids.species.castformrainy)) {
        return 'the rainy forme was never recorded';
      }
      if (!everHas(fixture, 0, 0, p => p.types.includes(ids.types.water))) {
        return 'the rainy forme was not Water-typed';
      }
      return null;
    },
  },
  {
    name: 'forecast_follows_the_sun_and_reverts_when_it_ends',
    p1: () => team(castform(), bulky()),
    p2: () => team(sunny()),
    script: [
      {p1: ['protect', 'protect'], p2: [{move: 'sunnyday'}, 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
      {p1: ['protect', 'protect'], p2: ['protect', 'protect']},
    ],
    coverage: {ability: 'forecast'},
    verify(fixture, session) {
      if (!logHas(session, /\|-formechange\|p1a: s0\|Castform-Sunny\|\[msg\]\|\[from\] ability: Forecast/)) {
        return 'the sunny forme was never announced';
      }
      if (!everHas(fixture, 0, 0, p => p.species === ids.species.castformsunny)) {
        return 'the sunny forme was never recorded';
      }
      // The weather ends five turns later and the handler set reverts the forme.
      const formes = fixture.steps.flatMap(step => step.expected.sides[0].pokemon
        .filter(p => p.roster === 0).map(p => p.species));
      if (!formes.some(species => species === ids.species.castform)) {
        return 'the forme never reverted after the weather ended';
      }
      return null;
    },
  },
  {
    name: 'forecast_control_without_weather',
    p1: () => team(castform(), bulky()),
    p2: () => foeTeam(),
    script: [{p1: ['protect', 'protect'], p2: ['protect', 'protect']}],
    coverage: {ability: 'forecast'},
    verify(fixture, session) {
      if (logHas(session, /\|-formechange\|p1a: s0\|/)) return 'the forme moved without weather';
      if (!fixture.steps.every(step => step.expected.sides[0].pokemon
        .filter(p => p.roster === 0).every(p => p.species === ids.species.castform))) {
        return 'a weather forme was recorded without weather';
      }
      return null;
    },
  },
];

runTrials(TRIALS, {seedBase: 8200, artifact: 'more_forecast.json', debugEnv: 'DEBUG_FORECAST'});
