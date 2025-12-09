import sys
sys.path.insert(0, 'c:\\Projects2025\\MLProject\\Project')

from src.backend import create_app

app = create_app()

with app.test_client() as client:
    response = client.get('/api/symbols')
    data = response.get_json()

    # Find BMO.TO
    bmo = [s for s in data if s['symbol'] == 'BMO.TO'][0]
    print('BMO.TO data from API:')
    for key in ['symbol', 'test_r2', 'test_rmse', 'test_direction_accuracy', 'test_mae']:
        print(f'  {key}: {bmo.get(key)}')
