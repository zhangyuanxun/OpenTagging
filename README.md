## OpenTagging for attribute value extraction from text

A Pytorch implementation of "Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title" (ACL 2019). [[pdf]](https://www.aclweb.org/anthology/P19-1514.pdf)

### Requirements:
* Pytorch>=1.9.0
* Python3.8

### Train and Evaluate
```
python src/main.py --do_train --do_eval --num_train_epochs 10
```

### Examples of outputs
```
{
        "context": "luontnor professional men soccer boots brand long spikes soccer shoes outdoor lawn mens football boots training chuteiras",
        "attribute": "types of hobnail",
        "true value": "long spikes",
        "predict value": "long spikes"
},
{
        "context": "150x hd protable astronomical telescope tripod powerful terrestrial space monocular telescope moon watching binoculars",
        "attribute": "type",
        "true value": "monocular",
        "predict value": "monocular"
},
```

### Model Serving
Model is served via a REST API with Flask,
```
python src/app.py
```

#### cURL request
```
curl -X POST http://0.0.0.0:8005/predict -H 'Content-Type: application/json' -d '{ "context": "Mens Womens Riding Cycling Socks Bicycle sports socks Breathable Anti-sweat Socks Basketball Football Socks", "attribute": "Hose Height"}'
```
#### Output
```
{"result":[{"context":"[CLS] mens womens riding cycling socks bicycle sports socks breathable anti - sweat socks basketball football socks [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]","position":[7,7],"tokens":["[CLS]","men","##s","women","##s","riding","cycling","socks","bicycle","sports","socks","breath","##able","anti","-","sweat","socks","basketball","football","socks","[SEP]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]","[PAD]"],"value":"socks"}]}
```
