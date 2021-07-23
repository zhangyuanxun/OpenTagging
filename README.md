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