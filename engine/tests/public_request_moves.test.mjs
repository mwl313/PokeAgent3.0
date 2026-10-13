import {test} from 'node:test';
import assert from 'node:assert/strict';
import {publicMoveDisabled,publicMoveTarget} from './public_request_moves.mjs';

test('hidden internal flag does not override the transmitted boolean',()=>{
  assert.equal(publicMoveDisabled({moves:[{id:'protect',disabled:false}]},{id:'protect',disabled:'hidden'}),false);
  assert.equal(publicMoveDisabled({moves:[{id:'protect',disabled:true}]},{id:'protect',disabled:false}),true);
});
test('identity uses IDs, not stale roster slot order',()=>{
  const request={moves:[{id:'curse',disabled:false,target:'self'},{id:'protect',disabled:true,target:'self'}]};
  assert.equal(publicMoveTarget(request,{id:'curse',target:'normal'}),'self');
  assert.equal(publicMoveDisabled(request,{id:'protect'}),true);
});
test('Struggle substitution never enables private original moves',()=>{
  assert.equal(publicMoveDisabled({moves:[{id:'struggle',disabled:false}]},{id:'protect'}),true);
});
test('locked singleton and recharge lists have exact identity',()=>{
  assert.equal(publicMoveDisabled({moves:[{id:'solarbeam'}]},{id:'solarbeam'}),false);
  assert.equal(publicMoveDisabled({moves:[{id:'recharge'}]},{id:'solarbeam'}),true);
});
test('hidden is not a valid transmitted disabled value',()=>{
  assert.throws(()=>publicMoveDisabled({moves:[{id:'protect',disabled:'hidden'}]},{id:'protect'}),/non-public/);
});
