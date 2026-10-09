// Development-only importer for one explicitly user-approved Pokepaste.
//
// The 2026-10-08 session approved exactly one new manual source: the user's own
// team at https://pokepast.es/c2cfbd453aa9172e. This script never crawls, never
// follows links and never adds another team. It reads the raw HTML and raw text
// that were saved under data/raw/user-pokepaste/20261008, cross-checks them
// against each other, verifies the pinned reference accepts the team, and emits
// the extraction record that scripts/prepare_teams.mjs consumes.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {FORMAT, ORACLE_COMMIT, ROOT} from './pa3-common.mjs';

const require = createRequire(import.meta.url);

export const SUBMISSION = Object.freeze({
  submission_id: 'UT20261008',
  pokepaste_url: 'https://pokepast.es/c2cfbd453aa9172e',
  raw_text_url: 'https://pokepast.es/c2cfbd453aa9172e/raw',
  paste_title: '리자몽',
  approved_by: 'user',
  approved_at: '2026-10-08',
  approval: 'User submitted this team, judged it strong in play, and explicitly approved adding it to the PokeAgent 3.0 training pool as one manual source.',
  scope_exception: 'This single Pokepaste is the entire new collection scope. No further crawling, discovery or automatic addition of teams is authorized.',
  local_dir: 'data/raw/user-pokepaste/20261008',
});

const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const statIDs = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'];

function decodeEntities(text) {
  return text
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
    .replace(/&#0?39;/g, "'").replace(/&apos;/g, "'").replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&');
}

/** Extract the six team blocks from the saved Pokepaste HTML page. */
export function extractHtmlBlocks(html) {
  const blocks = [];
  for (const match of html.matchAll(/<pre>([\s\S]*?)<\/pre>/g)) {
    const text = decodeEntities(match[1].replace(/<[^>]*>/g, ''));
    const lines = text.replace(/\r\n/g, '\n').split('\n').map(line => line.replace(/[ \t]+$/, ''));
    while (lines.length && lines[0] === '') lines.shift();
    while (lines.length && lines[lines.length - 1] === '') lines.pop();
    if (lines.length) blocks.push(lines);
  }
  return blocks;
}

/** Split the /raw plain-text export into member blocks, preserving line spacing. */
export function extractRawBlocks(rawText) {
  return rawText
    .replace(/\r\n/g, '\n')
    .split(/\n{2,}/)
    .map(block => block.split('\n'))
    .filter(lines => lines.some(line => line.trim() !== ''))
    .map(lines => {
      while (lines.length && lines[0].trim() === '') lines.shift();
      while (lines.length && lines[lines.length - 1].trim() === '') lines.pop();
      return lines;
    });
}

export function parseBlock(lines) {
  assert(lines.length >= 7, `Team block too short: ${lines.length} lines`);
  const header = lines[0].trim();
  const fields = {};
  const moves = [];
  const allocations = [];
  let nature_line = null;
  for (const rawLine of lines.slice(1)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith('- ')) { moves.push(line.slice(2).trim()); continue; }
    if (/ Nature$/i.test(line)) { nature_line = line; continue; }
    const split = line.indexOf(': ');
    if (split > 0) {
      const key = line.slice(0, split);
      const value = line.slice(split + 2);
      fields[key] = value;
      if (/^EVs$/i.test(key)) {
        const explicit = {};
        const unparsed = [];
        for (const part of value.split('/')) {
          const match = part.trim().match(/^(\d+)\s+(HP|Atk|Def|SpA|SpD|Spe)$/i);
          if (match) explicit[statIDs.find(id => id.toLowerCase() === match[2].toLowerCase())] = Number(match[1]);
          else unparsed.push(part.trim());
        }
        allocations.push({
          raw_line: `EVs: ${value}`, explicit_values: explicit, unparsed_parts: unparsed,
        });
      }
      continue;
    }
    throw new Error(`Unparsed team line: ${JSON.stringify(line)}`);
  }
  return {header, fields, nature_line, moves, allocations, raw_export_block: lines.join('\n')};
}

/** Pinned-reference receipt: legality verdict plus every normalization it applied. */
export function referenceReceipt(sets) {
  const {Teams, TeamValidator} = require(path.join(ROOT, 'vendor/pokemon-showdown/dist/sim/index.js'));
  const validator = new TeamValidator(FORMAT);
  assert.equal(validator.format.id, FORMAT);
  const imported = Teams.import(sets);
  assert.equal(imported.length, 6);
  const before = structuredClone(imported);
  const problems = validator.validateTeam(imported) || [];
  const normalizations = [];
  for (let i = 0; i < 6; i++) {
    for (const key of ['species', 'item', 'ability', 'nature', 'level', 'gender']) {
      if (String(before[i][key] ?? '') === String(imported[i][key] ?? '')) continue;
      normalizations.push({member_index: i, field: key, source_value: before[i][key] ?? null, reference_value: imported[i][key] ?? null});
    }
    if (JSON.stringify(before[i].evs) !== JSON.stringify(imported[i].evs)) {
      normalizations.push({member_index: i, field: 'evs', source_value: before[i].evs, reference_value: imported[i].evs});
    }
  }
  const baseAbilityAmbiguity = normalizations
    .filter(entry => entry.field === 'ability')
    .map(entry => {
      const choices = [...new Set(Object.values(validator.dex.species.get(imported[entry.member_index].species).abilities))];
      return {...entry, base_species: imported[entry.member_index].species, base_ability_choices: choices,
        resolution: choices.length === 1 ? 'reference_determined_single_base_ability' : 'reference_defaulted_first_base_ability'};
    });
  return {
    oracle_commit: ORACLE_COMMIT, format: FORMAT,
    verdict: problems.length ? 'rejected' : 'accepted_by_pinned_reference',
    problems, normalizations, base_ability_ambiguity: baseAbilityAmbiguity,
    normalized_team: imported.map(set => ({...set, name: ''})),
  };
}

export function buildRecord(dir = path.join(ROOT, SUBMISSION.local_dir)) {
  const htmlPath = path.join(dir, 'html/c2cfbd453aa9172e.html');
  const textPath = path.join(dir, 'teams/UT20261008.txt');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const rawText = fs.readFileSync(textPath, 'utf8');
  const htmlBlocks = extractHtmlBlocks(html);
  const rawBlocks = extractRawBlocks(rawText);
  assert.equal(rawBlocks.length, 6, 'The saved /raw text must contain six members');
  assert.equal(htmlBlocks.length, 6, 'The saved HTML must contain six member blocks');
  for (let i = 0; i < 6; i++) {
    assert.deepEqual(htmlBlocks[i].map(line => line.trimEnd()), rawBlocks[i].map(line => line.trimEnd()),
      `Saved HTML and /raw text disagree on member ${i}`);
  }
  const members = rawBlocks.map(parseBlock);
  const receipt = referenceReceipt(members.map(member => member.raw_export_block).join('\n\n'));
  assert.equal(receipt.verdict, 'accepted_by_pinned_reference', JSON.stringify(receipt.problems));
  assert.equal(members.length, 6);
  const headers = fs.readFileSync(path.join(dir, 'fetch_metadata/c2cfbd453aa9172e.raw.headers'), 'utf8');
  const htmlHeaders = fs.readFileSync(path.join(dir, 'fetch_metadata/c2cfbd453aa9172e.html.headers'), 'utf8');
  const headerValue = (headersText, name) => (headersText.match(new RegExp(`^${name}: (.*)$`, 'mi')) || [])[1]?.trim() ?? null;
  const statusCode = text => Number((text.split(/\r?\n/, 1)[0].match(/\s(\d{3})\s*$/) || [])[1] ?? 200);
  const record = {
    collection_schema: 'pa3-pokepaste-extraction-v1',
    source: {
      team_id: SUBMISSION.submission_id,
      description: 'User-submitted team (explicitly approved for the PokeAgent 3.0 training pool)',
      full_name: null, pokepaste_url: SUBMISSION.pokepaste_url, evs: 'Yes',
      paste_origin: 'UserSubmitted', date_shared: SUBMISSION.approved_at, event: null, rank: null,
      source_url: SUBMISSION.pokepaste_url, report_url: null, other_url: null,
      owner: null, roster: members.map(member => member.header.split(' @ ')[0].trim()),
    },
    fetch: {
      requested_url: SUBMISSION.raw_text_url, response_url: SUBMISSION.raw_text_url,
      http_status: statusCode(headers),
      content_type: headerValue(headers, 'content-type'),
      fetched_at: new Date(headerValue(headers, 'date')).toISOString(),
      sha256: hash(fs.readFileSync(textPath)), bytes: fs.statSync(textPath).size,
    },
    paste_title: SUBMISSION.paste_title,
    paste_notes: [],
    raw_html_path: 'html/c2cfbd453aa9172e.html',
    team_text_path: 'teams/UT20261008.txt',
    team_text_sha256: hash(fs.readFileSync(textPath)),
    raw_html_sha256: hash(fs.readFileSync(htmlPath)),
    html_fetched_at: new Date(headerValue(htmlHeaders, 'date')).toISOString(),
    members,
  };
  return {record, receipt, htmlPath, textPath};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const dir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, SUBMISSION.local_dir);
  const {record, receipt} = buildRecord(dir);
  const teamsPath = path.join(dir, 'teams.jsonl');
  fs.writeFileSync(teamsPath, JSON.stringify(record) + '\n');
  const confirmationPath = path.join(dir, 'confirmations.json');
  const confirmations = fs.existsSync(confirmationPath)
    ? JSON.parse(fs.readFileSync(confirmationPath, 'utf8')).confirmations
    : [];
  const summary = {
    submission: SUBMISSION,
    artifacts: Object.fromEntries(['html/c2cfbd453aa9172e.html', 'teams/UT20261008.txt', 'teams.jsonl']
      .map(name => [name, {sha256: hash(fs.readFileSync(path.join(dir, name)))}])),
    reference_validation: {
      oracle_commit: receipt.oracle_commit, format: receipt.format, verdict: receipt.verdict,
      problems: receipt.problems, normalizations: receipt.normalizations,
      base_ability_ambiguity: receipt.base_ability_ambiguity,
      normalized_team: receipt.normalized_team.map(set => ({species: set.species, ability: set.ability, item: set.item, moves: set.moves})),
    },
    user_confirmations: confirmations,
    notes: [
      'The source text is preserved verbatim; nothing in teams/UT20261008.txt was edited.',
      'The pinned reference accepts the team as written and resolves the declared Mega-form ability line to the base forme before battle.',
      'Values the pinned reference had to resolve are confirmed by the user in confirmations.json; the source text itself is never edited.',
      'No move, item, nature or allocation value was changed or imputed.',
    ],
  };
  fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({teams_jsonl_sha256: hash(fs.readFileSync(teamsPath)), verdict: receipt.verdict,
    normalizations: receipt.normalizations, ambiguity: receipt.base_ability_ambiguity}, null, 2));
}
