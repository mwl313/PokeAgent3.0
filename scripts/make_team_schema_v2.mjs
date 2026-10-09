// Derives data/schemas/team-record-v2.schema.json from the supplied
// pa3-team-v1 schema without touching the preserved specification package.
//
// The extension is deliberately narrow: it adds one provenance provider for the
// single user-approved manual Pokepaste and keeps every v1 constraint intact
// for VGCPastes records. Re-run with:
//   node scripts/make_team_schema_v2.mjs
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {ROOT} from './pa3-common.mjs';

export const SOURCE_SCHEMA = 'docs/spec/fullspec-1.1-minidc-20261006/PA3_TEAM_RECORD_SCHEMA.json';
export const TARGET_SCHEMA = 'data/schemas/team-record-v2.schema.json';

export function buildSchemaV2() {
  const base = JSON.parse(fs.readFileSync(path.join(ROOT, SOURCE_SCHEMA), 'utf8'));
  const schema = structuredClone(base);
  schema.title = 'PokeAgent3.0 accepted full-spread team record (v2 provenance extension)';
  schema.description = 'v1 plus exactly one provenance extension: the single user-approved manual Pokepaste source. ' +
    'Every VGCPastes constraint is unchanged. Raw rejects are stored separately; this schema does not itself establish legality or licensing.';
  schema.properties.schema_version = {enum: ['pa3-team-v1', 'pa3-team-v2']};
  schema.properties.source.properties.provider = {enum: ['VGCPastes', 'user_submitted_pokepaste']};
  schema.properties.source.properties.source_tabs.items.enum = ['Champions M-A', 'Champions M-B', 'Champions M-C', 'User Submission'];
  schema.properties.source.properties.submission = {
    type: 'object',
    required: ['approved_by', 'approved_at', 'approval', 'scope_exception'],
    properties: {
      approved_by: {type: 'string', minLength: 1},
      approved_at: {type: 'string', minLength: 1},
      approval: {type: 'string', minLength: 1},
      scope_exception: {type: 'string', minLength: 1},
    },
  };
  schema.properties.source.properties.ability_resolutions = {
    type: 'array',
    items: {
      type: 'object',
      required: ['member_index', 'source_value', 'reference_value', 'policy'],
      properties: {
        member_index: {type: 'integer', minimum: 0, maximum: 5},
        source_value: {type: 'string'},
        reference_value: {type: 'string'},
        policy: {enum: ['reference_validator_base_form_default', 'reference_determined_single_base_ability']},
      },
    },
  };
  schema.allOf = [
    {
      // A manual submission must declare its approval and use the v2 version tag.
      if: {properties: {source: {properties: {provider: {const: 'user_submitted_pokepaste'}}, required: ['provider']}}, required: ['source']},
      then: {
        properties: {
          schema_version: {const: 'pa3-team-v2'},
          source: {
            required: ['submission'],
            properties: {source_tabs: {type: 'array', minItems: 1, items: {const: 'User Submission'}}},
          },
        },
      },
    },
    {
      // Every VGCPastes record keeps the original tab vocabulary and the v1 tag.
      if: {properties: {source: {properties: {provider: {const: 'VGCPastes'}}, required: ['provider']}}, required: ['source']},
      then: {
        properties: {
          schema_version: {const: 'pa3-team-v1'},
          source: {
            properties: {
              source_tabs: {type: 'array', minItems: 1, items: {enum: ['Champions M-A', 'Champions M-B', 'Champions M-C']}},
            },
          },
        },
      },
    },
  ];
  schema['x-derivation'] = {source: SOURCE_SCHEMA, generator: 'scripts/make_team_schema_v2.mjs'};
  schema['x-semantic-validation'] = [...base['x-semantic-validation'],
    'user_submitted_pokepaste records are one explicitly approved manual source; they are never presented as VGCPastes collection records.',
    'A reference_validator_base_form_default ability resolution records both the declared Mega-form ability and the resolved base-form ability, and is flagged for user confirmation.'];
  return schema;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const schema = buildSchemaV2();
  const target = path.join(ROOT, TARGET_SCHEMA);
  const serialized = JSON.stringify(schema, null, 2) + '\n';
  if (fs.existsSync(target)) {
    assert_stable(fs.readFileSync(target, 'utf8'), serialized);
  } else {
    fs.writeFileSync(target, serialized);
  }
  console.log(JSON.stringify({target: TARGET_SCHEMA, source: SOURCE_SCHEMA,
    sha256: (await import('node:crypto')).createHash('sha256').update(serialized).digest('hex')}, null, 2));
}

function assert_stable(existing, next) {
  if (existing !== next) throw new Error('Existing v2 schema differs from the derived schema; inspect before overwriting');
}
