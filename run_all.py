"""Re-run the ten paired seeds used in Section 5.3 (20 training runs)."""
import json
from experiments import ROOT, train

for seed in range(20261125,20261135):
    for free in [False,True]:
        result=train(seed=seed,free_shape=free,steps=5000,batch=1024,h=.01)
        folder=ROOT/('free' if free else 'fixed')
        folder.mkdir(exist_ok=True)
        path=folder/f'seed_{seed}_h_0.01.json'
        path.write_text(json.dumps(result,indent=2))
        final=result['records'][-1]
        print(seed,'learned shape' if free else 'fixed shape',
              'KL',final['kl'],'value RRMSE',final['value_rrmse'],flush=True)
