# PPT learning

The learning framework proposed in [Gensim2](gensim2.github.io). 

## Requirements

```python
cd ppt_learning/third_party
git clone git@github.com:guochengqian/openpoints.git # For PointNext
cd ..
bash install.sh
pip install -e .
```

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