# OKIN
Opinion Knowledge Injection Network for Aspect Extraction
项目模型研发中。有需求，请联系13718537922。

## 训练及测试

Train the Restaurant model
```
python script/train.py --domain restaurant 
```
```
nohup python script/train.py --domain restaurant  > nohup_rest.out 2>&1 & 
nohup python script/train.py --domain restaurant & tail -f nohup.out
```

Train the Laptop model
```
python script/train.py 
```


Evaluate Restaurant dataset
```
python script/evaluation.py --domain restaurant 
```

```
nohup python script/train.py  > nohup.out 2>&1 & 
nohup python script/train.py & tail -f nohup.out
```

Evaluate Laptop dataset
```
python script/evaluation.py
```

test
```
nohup python test.py > nohup_test.out 2>&1 & 
```
