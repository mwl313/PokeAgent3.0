// Development fixture adapter. Never use raw moveSlot.disabled as publicity.
// The original roster move list is owner-known metadata, used only to retain
// stable slot/PP identity when the served list substitutes Struggle/recharge.
export function publicMoveDisabled(info, move) {
  // No active move request: the enclosing Wait/Replacement kind forbids a
  // move. Do not serialize a stale internal restriction as current knowledge.
  if (!info) return false;
  const served = info.moves?.find(entry => entry.id === move.id);
  if (!served) return true;
  if (served.disabled != null && typeof served.disabled !== 'boolean') {
    throw new Error(`non-public disabled value in served request: ${served.disabled}`);
  }
  return served.disabled === true;
}

export function publicMoveTarget(info, move) {
  return info?.moves?.find(entry => entry.id === move.id)?.target ?? move.target;
}
