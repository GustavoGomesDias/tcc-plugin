## TODO
Reunião - 30/08/23
- [ ] Criar um método que recebe o JSON (dataset) e uma medida de avaliação como parâmetro, e ele retorna as features e as labels
    - Label = sistema usado para gerar a descrição
    - Pegar o melhor sistema usando como base a medida de avaliação passada como parâmetro

```python
features, label = get_features(dataset, eval_measure='medida de avaliação', measure_type='tipo da medida, tendo f como padrão')
```