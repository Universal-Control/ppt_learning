# PPT learning

The learning framework proposed in [Gensim2](gensim2.github.io). 

## Run script
```python
python run.py domains=isaacsim
```


## Fix openpoints old version bug:
In `openpoints/transforms/point_transformer_gpu.py`:
```
282: if isinstance(self.angle, collections.Iterable):
```
Change to:

```
282: if isinstance(self.angle, collections.abc.Iterable):
```