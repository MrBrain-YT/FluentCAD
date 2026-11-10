"""Session timeline for solid snapshots and human-readable operation parameters."""
import math


class Timeline:
    def __init__(self):
        self.entries = []
        self.cursor = 0

    def append(self, title, parameters=None):
        record = validate_entry({'title': title, 'parameters': parameters or {}})
        self.entries = self.entries[:self.cursor]
        self.entries.append(record)
        self.cursor += 1

    def rename(self, index, title):
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError('Choose an operation to rename.')
        record = validate_entry({'title': title.strip(), 'parameters': self.entries[index]['parameters']})
        self.entries[index] = record

    def undo(self):
        self.cursor = max(0, self.cursor - 1)

    def redo(self):
        self.cursor = min(len(self.entries), self.cursor + 1)


def validate_entry(record):
    if not isinstance(record, dict) or not isinstance(record.get('title'), str) or not 0 < len(record['title']) <= 128:
        raise ValueError('Invalid timeline entry.')
    params = record.get('parameters', {})
    if not isinstance(params, dict) or len(params) > 64:
        raise ValueError('Invalid operation parameters.')
    result = {}
    for key, value in params.items():
        if not isinstance(key, str) or len(key) > 128 or not isinstance(value, (str, int, float, bool)):
            raise ValueError('Invalid operation parameter.')
        if isinstance(value, str) and len(value) > 512:
            raise ValueError('Operation parameter is too long.')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Operation parameter must be finite.')
        result[key] = value
    return {'title': record['title'], 'parameters': result}


def validate_timeline(data, undo_count, redo_count):
    if data is None:
        return {'entries': [{'title': 'Earlier operation', 'parameters': {}}
                            for _ in range(undo_count + redo_count)], 'cursor': undo_count}
    if not isinstance(data, dict) or not isinstance(data.get('entries'), list):
        raise ValueError('Invalid timeline.')
    entries = data['entries']
    cursor = data.get('cursor')
    if len(entries) != undo_count + redo_count or type(cursor) is not int or cursor != undo_count:
        raise ValueError('Timeline does not match saved undo history.')
    return {'entries': [validate_entry(entry) for entry in entries], 'cursor': cursor}
