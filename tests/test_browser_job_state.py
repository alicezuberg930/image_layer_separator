import json
import shutil
import subprocess
import unittest

from ldivider.ld_ui import READ_JOB_JS, SAVE_JOB_JS


class BrowserJobStateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is needed to check client JavaScript")
    def test_storage_restores_the_current_id_and_handles_unavailable_storage(self):
        script = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = JSON.parse(fs.readFileSync(0, 'utf8'));
const values = new Map();
let listeners = 0;
const context = {
  window: {},
  document: { addEventListener() { listeners++; } },
  localStorage: {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, value); },
    removeItem(key) { values.delete(key); }
  }
};
const read = vm.runInNewContext(source.read, context);
const save = vm.runInNewContext(source.save, context);
const stored = () => JSON.parse(JSON.stringify(read()));
assert.deepEqual(stored(), ['']);
save('a'.repeat(32));
assert.deepEqual(stored(), ['a'.repeat(32)]);
save('b'.repeat(32));
assert.deepEqual(stored(), ['b'.repeat(32)]);
assert.equal(listeners, 1);
save('');
assert.deepEqual(stored(), ['']);
context.localStorage.getItem = () => { throw Error('Storage unavailable'); };
context.localStorage.setItem = () => { throw Error('Storage unavailable'); };
assert.deepEqual(stored(), ['']);
save('c'.repeat(32));
console.log('Browser job storage checks passed.');
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            input=json.dumps({"read": READ_JOB_JS, "save": SAVE_JOB_JS}),
            capture_output=True, text=True, timeout=10, check=True,
        )
        self.assertIn("checks passed", result.stdout)
