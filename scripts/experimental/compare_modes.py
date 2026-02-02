import base64
import statistics
import time

from app import app

IMAGE_PATH = '/Users/awakawanaoki/Downloads/IMG_0231.JPG'
RUNS = 3


def metrics(data):
    mitori = next((s for s in data.get('sections', []) if 'みとり' in str(s.get('title', ''))), None)
    kake = next((s for s in data.get('sections', []) if 'かけざん' in str(s.get('title', ''))), None)

    lens = {i: 0 for i in range(1, 9)}
    if mitori and isinstance(mitori.get('results'), list):
        for r in mitori.get('results', []):
            try:
                c = int(r.get('column'))
            except Exception:
                continue
            nums = r.get('numbers', [])
            lens[c] = len(nums) if isinstance(nums, list) else 0

    expected = {1: 7, 2: 7, 3: 7, 4: 7, 5: 8, 6: 8, 7: 8, 8: 8}
    mitori_exact = sum(1 for c, n in expected.items() if lens.get(c, 0) == n)

    kake_items = 0
    kake_3x3 = 0
    if kake and isinstance(kake.get('items'), list):
        items = kake.get('items', [])
        kake_items = len(items)
        for it in items:
            expr = str(it.get('expression', '')).replace('×', '*').replace('x', '*').replace('X', '*')
            parts = [p for p in expr.split('*') if p]
            if len(parts) >= 2 and len(parts[0].strip()) == 3 and len(parts[1].strip()) == 3:
                kake_3x3 += 1

    score = mitori_exact * 10 + min(kake_items, 6) * 2 + kake_3x3 * 6
    return {
        'mitori_exact_cols': mitori_exact,
        'kake_items': kake_items,
        'kake_3x3': kake_3x3,
        'acc_score': score,
        'mitori_lens': [lens[i] for i in range(1, 9)],
    }


def run_mode(mode, payload_image):
    client = app.test_client()
    wall_times = []
    model_times = []
    metric_list = []
    for _ in range(RUNS):
        t0 = time.time()
        resp = client.post('/ocr', json={'image': payload_image, 'mode': mode})
        wall = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            raise RuntimeError(f"{mode} failed: {resp.status_code} {resp.get_data(as_text=True)[:300]}")
        data = resp.get_json()
        wall_times.append(wall)
        model_times.append(int(data.get('processing_time_ms', wall)))
        metric_list.append(metrics(data))

    avg = {
        'mode': mode,
        'avg_wall_ms': int(statistics.mean(wall_times)),
        'avg_processing_ms': int(statistics.mean(model_times)),
        'avg_acc_score': round(statistics.mean([m['acc_score'] for m in metric_list]), 2),
        'mitori_exact_cols': [m['mitori_exact_cols'] for m in metric_list],
        'kake_3x3': [m['kake_3x3'] for m in metric_list],
        'mitori_lens_samples': [m['mitori_lens'] for m in metric_list],
    }
    return avg


def main():
    with open(IMAGE_PATH, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    payload_image = 'data:image/jpeg;base64,' + b64

    a = run_mode('accuracy', payload_image)
    s = run_mode('speed', payload_image)

    print('ACCURACY:', a)
    print('SPEED   :', s)

    best = a
    if s['avg_acc_score'] >= a['avg_acc_score'] and s['avg_processing_ms'] < a['avg_processing_ms']:
        best = s
    print('BEST:', best['mode'])


if __name__ == '__main__':
    main()
