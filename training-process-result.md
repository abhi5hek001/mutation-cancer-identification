(myenv-bi) oshu@Abhishek:/mnt/e/College/Sem6/BI/cancer genome sequence prediction$ python training_updated.py
Data shape: (53811, 11)
Columns: ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'Variant_Classification', 'Variant_Type', 'Strand', 'Genomic_Context_Sequence', 'Cancer_Type']
Mutation type distribution:
Variant_Classification
Silent               18320
Nonsense_Mutation    18157
Missense_Mutation    17334
Name: count, dtype: int64
Cancer type distribution:
Cancer_Type
BLCA    20667
LUSC    20645
KIRC    12499
Name: count, dtype: int64
Mutation classes: ['Missense_Mutation' 'Nonsense_Mutation' 'Silent']
Cancer classes: ['BLCA' 'KIRC' 'LUSC']
Number of mutation classes: 3
Number of cancer classes: 3

Fold 1/5
Initial samples: 34438, Valid samples: 34438
Initial samples: 8610, Valid samples: 8610
Initial samples: 10763, Valid samples: 10763
Using device: cpu
Epoch 1/20: Train Loss: 2.0782, Mutation Acc: 0.4564, Cancer Acc: 0.5008, Val Loss: 2.0516, Val Mutation Acc: 0.4765, Val Cancer Acc: 0.5163
Epoch 2/20: Train Loss: 2.0632, Mutation Acc: 0.4701, Cancer Acc: 0.5077, Val Loss: 2.0536, Val Mutation Acc: 0.4796, Val Cancer Acc: 0.5161
Epoch 3/20: Train Loss: 2.0594, Mutation Acc: 0.4733, Cancer Acc: 0.5087, Val Loss: 2.0429, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5160
Epoch 4/20: Train Loss: 2.0596, Mutation Acc: 0.4731, Cancer Acc: 0.5079, Val Loss: 2.0445, Val Mutation Acc: 0.4799, Val Cancer Acc: 0.5163
Epoch 5/20: Train Loss: 2.0548, Mutation Acc: 0.4745, Cancer Acc: 0.5093, Val Loss: 2.0489, Val Mutation Acc: 0.4653, Val Cancer Acc: 0.5163
Epoch 6/20: Train Loss: 2.0550, Mutation Acc: 0.4732, Cancer Acc: 0.5096, Val Loss: 2.0456, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5161
Epoch 7/20: Train Loss: 2.0543, Mutation Acc: 0.4757, Cancer Acc: 0.5089, Val Loss: 2.0400, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5163
Epoch 8/20: Train Loss: 2.0549, Mutation Acc: 0.4760, Cancer Acc: 0.5100, Val Loss: 2.0430, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5163
Epoch 9/20: Train Loss: 2.0522, Mutation Acc: 0.4749, Cancer Acc: 0.5081, Val Loss: 2.0425, Val Mutation Acc: 0.4801, Val Cancer Acc: 0.5163
Epoch 10/20: Train Loss: 2.0522, Mutation Acc: 0.4759, Cancer Acc: 0.5099, Val Loss: 2.0435, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5159
Epoch 11/20: Train Loss: 2.0504, Mutation Acc: 0.4752, Cancer Acc: 0.5097, Val Loss: 2.0415, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5163
Epoch 12/20: Train Loss: 2.0500, Mutation Acc: 0.4763, Cancer Acc: 0.5094, Val Loss: 2.0402, Val Mutation Acc: 0.4805, Val Cancer Acc: 0.5163
Epoch 13/20: Train Loss: 2.0506, Mutation Acc: 0.4760, Cancer Acc: 0.5095, Val Loss: 2.0388, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5163
Epoch 14/20: Train Loss: 2.0499, Mutation Acc: 0.4754, Cancer Acc: 0.5100, Val Loss: 2.0391, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5163
Epoch 15/20: Train Loss: 2.0504, Mutation Acc: 0.4762, Cancer Acc: 0.5102, Val Loss: 2.0441, Val Mutation Acc: 0.4805, Val Cancer Acc: 0.5165
Epoch 16/20: Train Loss: 2.0501, Mutation Acc: 0.4763, Cancer Acc: 0.5099, Val Loss: 2.0370, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5164
Epoch 17/20: Train Loss: 2.0504, Mutation Acc: 0.4761, Cancer Acc: 0.5099, Val Loss: 2.0414, Val Mutation Acc: 0.4798, Val Cancer Acc: 0.5165
Epoch 18/20: Train Loss: 2.0486, Mutation Acc: 0.4757, Cancer Acc: 0.5106, Val Loss: 2.0353, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5165
Epoch 19/20: Train Loss: 2.0487, Mutation Acc: 0.4766, Cancer Acc: 0.5098, Val Loss: 2.0379, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5164
Epoch 20/20: Train Loss: 2.0486, Mutation Acc: 0.4752, Cancer Acc: 0.5099, Val Loss: 2.0417, Val Mutation Acc: 0.4804, Val Cancer Acc: 0.5160
Mutation Classification Accuracy: 0.4613
Mutation Classification Report:
                   precision    recall  f1-score   support

Missense_Mutation       0.47      0.23      0.31      3541
Nonsense_Mutation       0.49      0.51      0.50      3635
           Silent       0.44      0.64      0.52      3587

         accuracy                           0.46     10763
        macro avg       0.47      0.46      0.44     10763
     weighted avg       0.47      0.46      0.44     10763

Cancer Type Classification Accuracy: 0.5109
Cancer Type Classification Report:
              precision    recall  f1-score   support

        BLCA       0.50      0.72      0.59      4134
        KIRC       0.51      0.21      0.30      2500
        LUSC       0.53      0.49      0.51      4129

    accuracy                           0.51     10763
   macro avg       0.51      0.47      0.47     10763
weighted avg       0.51      0.51      0.49     10763


Fold 2/5
Initial samples: 34439, Valid samples: 34439
Initial samples: 8610, Valid samples: 8610
Initial samples: 10762, Valid samples: 10762
Using device: cpu
Epoch 1/20: Train Loss: 2.0809, Mutation Acc: 0.4552, Cancer Acc: 0.4985, Val Loss: 2.0552, Val Mutation Acc: 0.4668, Val Cancer Acc: 0.5116
Epoch 2/20: Train Loss: 2.0647, Mutation Acc: 0.4679, Cancer Acc: 0.5065, Val Loss: 2.0583, Val Mutation Acc: 0.4675, Val Cancer Acc: 0.5100
Epoch 3/20: Train Loss: 2.0606, Mutation Acc: 0.4663, Cancer Acc: 0.5058, Val Loss: 2.0523, Val Mutation Acc: 0.4704, Val Cancer Acc: 0.5153
Epoch 4/20: Train Loss: 2.0583, Mutation Acc: 0.4703, Cancer Acc: 0.5083, Val Loss: 2.0484, Val Mutation Acc: 0.4628, Val Cancer Acc: 0.5156
Epoch 5/20: Train Loss: 2.0587, Mutation Acc: 0.4681, Cancer Acc: 0.5074, Val Loss: 2.0663, Val Mutation Acc: 0.4610, Val Cancer Acc: 0.5099
Epoch 6/20: Train Loss: 2.0576, Mutation Acc: 0.4690, Cancer Acc: 0.5083, Val Loss: 2.0496, Val Mutation Acc: 0.4700, Val Cancer Acc: 0.5152
Epoch 7/20: Train Loss: 2.0558, Mutation Acc: 0.4713, Cancer Acc: 0.5099, Val Loss: 2.0509, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5156
Epoch 8/20: Train Loss: 2.0550, Mutation Acc: 0.4699, Cancer Acc: 0.5101, Val Loss: 2.0508, Val Mutation Acc: 0.4699, Val Cancer Acc: 0.5154
Epoch 9/20: Train Loss: 2.0536, Mutation Acc: 0.4713, Cancer Acc: 0.5088, Val Loss: 2.0462, Val Mutation Acc: 0.4702, Val Cancer Acc: 0.5158
Epoch 10/20: Train Loss: 2.0525, Mutation Acc: 0.4721, Cancer Acc: 0.5097, Val Loss: 2.0464, Val Mutation Acc: 0.4697, Val Cancer Acc: 0.5157
Epoch 11/20: Train Loss: 2.0524, Mutation Acc: 0.4714, Cancer Acc: 0.5097, Val Loss: 2.0499, Val Mutation Acc: 0.4699, Val Cancer Acc: 0.5154
Epoch 12/20: Train Loss: 2.0521, Mutation Acc: 0.4718, Cancer Acc: 0.5091, Val Loss: 2.0457, Val Mutation Acc: 0.4700, Val Cancer Acc: 0.5159
Epoch 13/20: Train Loss: 2.0526, Mutation Acc: 0.4723, Cancer Acc: 0.5098, Val Loss: 2.0505, Val Mutation Acc: 0.4704, Val Cancer Acc: 0.5154
Epoch 14/20: Train Loss: 2.0517, Mutation Acc: 0.4721, Cancer Acc: 0.5088, Val Loss: 2.0498, Val Mutation Acc: 0.4700, Val Cancer Acc: 0.5154
Epoch 15/20: Train Loss: 2.0517, Mutation Acc: 0.4716, Cancer Acc: 0.5088, Val Loss: 2.0486, Val Mutation Acc: 0.4697, Val Cancer Acc: 0.5158
Epoch 16/20: Train Loss: 2.0500, Mutation Acc: 0.4724, Cancer Acc: 0.5092, Val Loss: 2.0522, Val Mutation Acc: 0.4696, Val Cancer Acc: 0.5158
Epoch 17/20: Train Loss: 2.0513, Mutation Acc: 0.4722, Cancer Acc: 0.5081, Val Loss: 2.0480, Val Mutation Acc: 0.4696, Val Cancer Acc: 0.5157
Epoch 18/20: Train Loss: 2.0503, Mutation Acc: 0.4719, Cancer Acc: 0.5095, Val Loss: 2.0472, Val Mutation Acc: 0.4698, Val Cancer Acc: 0.5158
Epoch 19/20: Train Loss: 2.0466, Mutation Acc: 0.4728, Cancer Acc: 0.5106, Val Loss: 2.0450, Val Mutation Acc: 0.4697, Val Cancer Acc: 0.5158
Epoch 20/20: Train Loss: 2.0459, Mutation Acc: 0.4728, Cancer Acc: 0.5101, Val Loss: 2.0432, Val Mutation Acc: 0.4696, Val Cancer Acc: 0.5157
Mutation Classification Accuracy: 0.4817
Mutation Classification Report:
                   precision    recall  f1-score   support

Missense_Mutation       0.49      0.24      0.32      3504
Nonsense_Mutation       0.51      0.55      0.53      3594
           Silent       0.46      0.65      0.54      3664

         accuracy                           0.48     10762
        macro avg       0.48      0.48      0.46     10762
     weighted avg       0.48      0.48      0.46     10762

Cancer Type Classification Accuracy: 0.5136
Cancer Type Classification Report:
              precision    recall  f1-score   support

        BLCA       0.51      0.72      0.59      4134
        KIRC       0.50      0.21      0.29      2499
        LUSC       0.53      0.50      0.51      4129

    accuracy                           0.51     10762
   macro avg       0.51      0.47      0.47     10762
weighted avg       0.51      0.51      0.49     10762


Fold 3/5
Initial samples: 34439, Valid samples: 34439
Initial samples: 8610, Valid samples: 8610
Initial samples: 10762, Valid samples: 10762
Using device: cpu
Epoch 1/20: Train Loss: 2.0784, Mutation Acc: 0.4587, Cancer Acc: 0.4957, Val Loss: 2.0541, Val Mutation Acc: 0.4634, Val Cancer Acc: 0.5111
Epoch 2/20: Train Loss: 2.0631, Mutation Acc: 0.4686, Cancer Acc: 0.5063, Val Loss: 2.0516, Val Mutation Acc: 0.4634, Val Cancer Acc: 0.5168
Epoch 3/20: Train Loss: 2.0601, Mutation Acc: 0.4718, Cancer Acc: 0.5072, Val Loss: 2.0547, Val Mutation Acc: 0.4661, Val Cancer Acc: 0.5168
Epoch 4/20: Train Loss: 2.0597, Mutation Acc: 0.4705, Cancer Acc: 0.5074, Val Loss: 2.0528, Val Mutation Acc: 0.4666, Val Cancer Acc: 0.5168
Epoch 5/20: Train Loss: 2.0585, Mutation Acc: 0.4722, Cancer Acc: 0.5075, Val Loss: 2.0463, Val Mutation Acc: 0.4666, Val Cancer Acc: 0.5167
Epoch 6/20: Train Loss: 2.0568, Mutation Acc: 0.4727, Cancer Acc: 0.5074, Val Loss: 2.0529, Val Mutation Acc: 0.4659, Val Cancer Acc: 0.5167
Epoch 7/20: Train Loss: 2.0555, Mutation Acc: 0.4728, Cancer Acc: 0.5084, Val Loss: 2.0497, Val Mutation Acc: 0.4625, Val Cancer Acc: 0.5167
Epoch 8/20: Train Loss: 2.0542, Mutation Acc: 0.4727, Cancer Acc: 0.5079, Val Loss: 2.0537, Val Mutation Acc: 0.4667, Val Cancer Acc: 0.5167
Epoch 9/20: Train Loss: 2.0538, Mutation Acc: 0.4733, Cancer Acc: 0.5086, Val Loss: 2.0492, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5167
Epoch 10/20: Train Loss: 2.0523, Mutation Acc: 0.4723, Cancer Acc: 0.5089, Val Loss: 2.0444, Val Mutation Acc: 0.4663, Val Cancer Acc: 0.5167
Epoch 11/20: Train Loss: 2.0523, Mutation Acc: 0.4739, Cancer Acc: 0.5083, Val Loss: 2.0490, Val Mutation Acc: 0.4662, Val Cancer Acc: 0.5167
Epoch 12/20: Train Loss: 2.0511, Mutation Acc: 0.4745, Cancer Acc: 0.5088, Val Loss: 2.0490, Val Mutation Acc: 0.4668, Val Cancer Acc: 0.5056
Epoch 13/20: Train Loss: 2.0529, Mutation Acc: 0.4738, Cancer Acc: 0.5081, Val Loss: 2.0442, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5166
Epoch 14/20: Train Loss: 2.0511, Mutation Acc: 0.4739, Cancer Acc: 0.5073, Val Loss: 2.0463, Val Mutation Acc: 0.4663, Val Cancer Acc: 0.5108
Epoch 15/20: Train Loss: 2.0506, Mutation Acc: 0.4736, Cancer Acc: 0.5087, Val Loss: 2.0438, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5167
Epoch 16/20: Train Loss: 2.0503, Mutation Acc: 0.4740, Cancer Acc: 0.5082, Val Loss: 2.0437, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5167
Epoch 17/20: Train Loss: 2.0504, Mutation Acc: 0.4733, Cancer Acc: 0.5082, Val Loss: 2.0456, Val Mutation Acc: 0.4663, Val Cancer Acc: 0.5049
Epoch 18/20: Train Loss: 2.0492, Mutation Acc: 0.4743, Cancer Acc: 0.5088, Val Loss: 2.0481, Val Mutation Acc: 0.4630, Val Cancer Acc: 0.5166
Epoch 19/20: Train Loss: 2.0506, Mutation Acc: 0.4738, Cancer Acc: 0.5092, Val Loss: 2.0436, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5167
Epoch 20/20: Train Loss: 2.0498, Mutation Acc: 0.4735, Cancer Acc: 0.5090, Val Loss: 2.0421, Val Mutation Acc: 0.4664, Val Cancer Acc: 0.5123
Mutation Classification Accuracy: 0.4789
Mutation Classification Report:
                   precision    recall  f1-score   support

Missense_Mutation       0.47      0.24      0.31      3411
Nonsense_Mutation       0.50      0.53      0.51      3627
           Silent       0.47      0.65      0.54      3724

         accuracy                           0.48     10762
        macro avg       0.48      0.47      0.46     10762
     weighted avg       0.48      0.48      0.46     10762

Cancer Type Classification Accuracy: 0.5149
Cancer Type Classification Report:
              precision    recall  f1-score   support

        BLCA       0.50      0.72      0.59      4133
        KIRC       0.50      0.25      0.33      2500
        LUSC       0.54      0.47      0.50      4129

    accuracy                           0.51     10762
   macro avg       0.51      0.48      0.48     10762
weighted avg       0.52      0.51      0.50     10762


Fold 4/5
Initial samples: 34439, Valid samples: 34439
Initial samples: 8610, Valid samples: 8610
Initial samples: 10762, Valid samples: 10762
Using device: cpu
Epoch 1/20: Train Loss: 2.0783, Mutation Acc: 0.4568, Cancer Acc: 0.5013, Val Loss: 2.0632, Val Mutation Acc: 0.4667, Val Cancer Acc: 0.5095
Epoch 2/20: Train Loss: 2.0608, Mutation Acc: 0.4664, Cancer Acc: 0.5093, Val Loss: 2.0679, Val Mutation Acc: 0.4695, Val Cancer Acc: 0.5124
Epoch 3/20: Train Loss: 2.0599, Mutation Acc: 0.4700, Cancer Acc: 0.5100, Val Loss: 2.0582, Val Mutation Acc: 0.4654, Val Cancer Acc: 0.5127
Epoch 4/20: Train Loss: 2.0606, Mutation Acc: 0.4661, Cancer Acc: 0.5094, Val Loss: 2.0532, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5128
Epoch 5/20: Train Loss: 2.0556, Mutation Acc: 0.4714, Cancer Acc: 0.5108, Val Loss: 2.0513, Val Mutation Acc: 0.4657, Val Cancer Acc: 0.5127
Epoch 6/20: Train Loss: 2.0527, Mutation Acc: 0.4723, Cancer Acc: 0.5101, Val Loss: 2.0541, Val Mutation Acc: 0.4697, Val Cancer Acc: 0.5128
Epoch 7/20: Train Loss: 2.0539, Mutation Acc: 0.4709, Cancer Acc: 0.5107, Val Loss: 2.0567, Val Mutation Acc: 0.4691, Val Cancer Acc: 0.5128
Epoch 8/20: Train Loss: 2.0539, Mutation Acc: 0.4716, Cancer Acc: 0.5104, Val Loss: 2.0537, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5128
Epoch 9/20: Train Loss: 2.0518, Mutation Acc: 0.4705, Cancer Acc: 0.5103, Val Loss: 2.0509, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5128
Epoch 10/20: Train Loss: 2.0516, Mutation Acc: 0.4727, Cancer Acc: 0.5105, Val Loss: 2.0485, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5127
Epoch 11/20: Train Loss: 2.0514, Mutation Acc: 0.4733, Cancer Acc: 0.5109, Val Loss: 2.0484, Val Mutation Acc: 0.4693, Val Cancer Acc: 0.5127
Epoch 12/20: Train Loss: 2.0513, Mutation Acc: 0.4727, Cancer Acc: 0.5103, Val Loss: 2.0526, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5127
Epoch 13/20: Train Loss: 2.0506, Mutation Acc: 0.4721, Cancer Acc: 0.5104, Val Loss: 2.0544, Val Mutation Acc: 0.4693, Val Cancer Acc: 0.5128
Epoch 14/20: Train Loss: 2.0511, Mutation Acc: 0.4734, Cancer Acc: 0.5114, Val Loss: 2.0535, Val Mutation Acc: 0.4668, Val Cancer Acc: 0.5127
Epoch 15/20: Train Loss: 2.0508, Mutation Acc: 0.4735, Cancer Acc: 0.5111, Val Loss: 2.0502, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5127
Epoch 16/20: Train Loss: 2.0495, Mutation Acc: 0.4738, Cancer Acc: 0.5108, Val Loss: 2.0505, Val Mutation Acc: 0.4691, Val Cancer Acc: 0.5127
Epoch 17/20: Train Loss: 2.0457, Mutation Acc: 0.4747, Cancer Acc: 0.5120, Val Loss: 2.0483, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5127
Epoch 18/20: Train Loss: 2.0452, Mutation Acc: 0.4745, Cancer Acc: 0.5118, Val Loss: 2.0492, Val Mutation Acc: 0.4693, Val Cancer Acc: 0.5127
Epoch 19/20: Train Loss: 2.0454, Mutation Acc: 0.4738, Cancer Acc: 0.5118, Val Loss: 2.0486, Val Mutation Acc: 0.4692, Val Cancer Acc: 0.5124
Epoch 20/20: Train Loss: 2.0451, Mutation Acc: 0.4746, Cancer Acc: 0.5118, Val Loss: 2.0504, Val Mutation Acc: 0.4689, Val Cancer Acc: 0.5125
Mutation Classification Accuracy: 0.4786
Mutation Classification Report:
                   precision    recall  f1-score   support

Missense_Mutation       0.47      0.24      0.32      3425
Nonsense_Mutation       0.50      0.53      0.51      3620
           Silent       0.47      0.65      0.54      3717

         accuracy                           0.48     10762
        macro avg       0.48      0.47      0.46     10762
     weighted avg       0.48      0.48      0.46     10762

Cancer Type Classification Accuracy: 0.5111
Cancer Type Classification Report:
              precision    recall  f1-score   support

        BLCA       0.51      0.71      0.59      4133
        KIRC       0.48      0.21      0.29      2500
        LUSC       0.53      0.49      0.51      4129

    accuracy                           0.51     10762
   macro avg       0.51      0.47      0.46     10762
weighted avg       0.51      0.51      0.49     10762


Fold 5/5
Initial samples: 34439, Valid samples: 34439
Initial samples: 8610, Valid samples: 8610
Initial samples: 10762, Valid samples: 10762
Using device: cpu
Epoch 1/20: Train Loss: 2.0770, Mutation Acc: 0.4554, Cancer Acc: 0.4992, Val Loss: 2.0518, Val Mutation Acc: 0.4718, Val Cancer Acc: 0.5137
Epoch 2/20: Train Loss: 2.0606, Mutation Acc: 0.4705, Cancer Acc: 0.5077, Val Loss: 2.0510, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5136
Epoch 3/20: Train Loss: 2.0594, Mutation Acc: 0.4703, Cancer Acc: 0.5090, Val Loss: 2.0560, Val Mutation Acc: 0.4729, Val Cancer Acc: 0.5135
Epoch 4/20: Train Loss: 2.0574, Mutation Acc: 0.4698, Cancer Acc: 0.5099, Val Loss: 2.0544, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5135
Epoch 5/20: Train Loss: 2.0574, Mutation Acc: 0.4717, Cancer Acc: 0.5106, Val Loss: 2.0517, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5134
Epoch 6/20: Train Loss: 2.0559, Mutation Acc: 0.4710, Cancer Acc: 0.5116, Val Loss: 2.0464, Val Mutation Acc: 0.4755, Val Cancer Acc: 0.5135
Epoch 7/20: Train Loss: 2.0542, Mutation Acc: 0.4712, Cancer Acc: 0.5105, Val Loss: 2.0520, Val Mutation Acc: 0.4748, Val Cancer Acc: 0.5135
Epoch 8/20: Train Loss: 2.0545, Mutation Acc: 0.4726, Cancer Acc: 0.5113, Val Loss: 2.0455, Val Mutation Acc: 0.4750, Val Cancer Acc: 0.5135
Epoch 9/20: Train Loss: 2.0518, Mutation Acc: 0.4728, Cancer Acc: 0.5123, Val Loss: 2.0521, Val Mutation Acc: 0.4749, Val Cancer Acc: 0.5135
Epoch 10/20: Train Loss: 2.0519, Mutation Acc: 0.4728, Cancer Acc: 0.5116, Val Loss: 2.0482, Val Mutation Acc: 0.4734, Val Cancer Acc: 0.5135
Epoch 11/20: Train Loss: 2.0514, Mutation Acc: 0.4733, Cancer Acc: 0.5107, Val Loss: 2.0464, Val Mutation Acc: 0.4715, Val Cancer Acc: 0.5135
Epoch 12/20: Train Loss: 2.0512, Mutation Acc: 0.4726, Cancer Acc: 0.5112, Val Loss: 2.0468, Val Mutation Acc: 0.4750, Val Cancer Acc: 0.5135
Epoch 13/20: Train Loss: 2.0503, Mutation Acc: 0.4742, Cancer Acc: 0.5110, Val Loss: 2.0471, Val Mutation Acc: 0.4731, Val Cancer Acc: 0.5132
Epoch 14/20: Train Loss: 2.0505, Mutation Acc: 0.4731, Cancer Acc: 0.5118, Val Loss: 2.0438, Val Mutation Acc: 0.4750, Val Cancer Acc: 0.5135
Epoch 15/20: Train Loss: 2.0503, Mutation Acc: 0.4741, Cancer Acc: 0.5114, Val Loss: 2.0489, Val Mutation Acc: 0.4731, Val Cancer Acc: 0.5132
Epoch 16/20: Train Loss: 2.0500, Mutation Acc: 0.4739, Cancer Acc: 0.5116, Val Loss: 2.0452, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5137
Epoch 17/20: Train Loss: 2.0481, Mutation Acc: 0.4744, Cancer Acc: 0.5118, Val Loss: 2.0509, Val Mutation Acc: 0.4655, Val Cancer Acc: 0.5137
Epoch 18/20: Train Loss: 2.0484, Mutation Acc: 0.4739, Cancer Acc: 0.5112, Val Loss: 2.0444, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5130
Epoch 19/20: Train Loss: 2.0477, Mutation Acc: 0.4734, Cancer Acc: 0.5115, Val Loss: 2.0442, Val Mutation Acc: 0.4748, Val Cancer Acc: 0.5137
Epoch 20/20: Train Loss: 2.0480, Mutation Acc: 0.4727, Cancer Acc: 0.5122, Val Loss: 2.0488, Val Mutation Acc: 0.4753, Val Cancer Acc: 0.5139
Mutation Classification Accuracy: 0.4722
Mutation Classification Report:
                   precision    recall  f1-score   support

Missense_Mutation       0.48      0.24      0.32      3453
Nonsense_Mutation       0.50      0.51      0.51      3681
           Silent       0.45      0.66      0.54      3628

         accuracy                           0.47     10762
        macro avg       0.48      0.47      0.45     10762
     weighted avg       0.48      0.47      0.45     10762

Cancer Type Classification Accuracy: 0.5095
Cancer Type Classification Report:
              precision    recall  f1-score   support

        BLCA       0.50      0.71      0.59      4133
        KIRC       0.52      0.23      0.31      2500
        LUSC       0.52      0.48      0.50      4129

    accuracy                           0.51     10762
   macro avg       0.51      0.47      0.47     10762
weighted avg       0.51      0.51      0.49     10762


Cross-Validation Results:
Fold 1: Mutation Accuracy = 0.4613, Cancer Accuracy = 0.5109
Fold 2: Mutation Accuracy = 0.4817, Cancer Accuracy = 0.5136
Fold 3: Mutation Accuracy = 0.4789, Cancer Accuracy = 0.5149
Fold 4: Mutation Accuracy = 0.4786, Cancer Accuracy = 0.5111
Fold 5: Mutation Accuracy = 0.4722, Cancer Accuracy = 0.5095
Average Mutation Accuracy: 0.4746 ± 0.0073
Average Cancer Accuracy: 0.5120 ± 0.0020

Training final model on the full dataset...
Initial samples: 43048, Valid samples: 43048
Initial samples: 10763, Valid samples: 10763
Using device: cpu
Epoch 1/20: Train Loss: 2.0754, Mutation Acc: 0.4598, Cancer Acc: 0.5020, Val Loss: 2.0605, Val Mutation Acc: 0.4617, Val Cancer Acc: 0.5139
Epoch 2/20: Train Loss: 2.0606, Mutation Acc: 0.4701, Cancer Acc: 0.5079, Val Loss: 2.0498, Val Mutation Acc: 0.4686, Val Cancer Acc: 0.5139
Epoch 3/20: Train Loss: 2.0612, Mutation Acc: 0.4700, Cancer Acc: 0.5082, Val Loss: 2.0556, Val Mutation Acc: 0.4667, Val Cancer Acc: 0.5139
Epoch 4/20: Train Loss: 2.0592, Mutation Acc: 0.4718, Cancer Acc: 0.5096, Val Loss: 2.0523, Val Mutation Acc: 0.4663, Val Cancer Acc: 0.5139
Epoch 5/20: Train Loss: 2.0552, Mutation Acc: 0.4729, Cancer Acc: 0.5098, Val Loss: 2.0484, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5136
Epoch 6/20: Train Loss: 2.0535, Mutation Acc: 0.4730, Cancer Acc: 0.5104, Val Loss: 2.0511, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5136
Epoch 7/20: Train Loss: 2.0520, Mutation Acc: 0.4741, Cancer Acc: 0.5097, Val Loss: 2.0537, Val Mutation Acc: 0.4671, Val Cancer Acc: 0.5138
Epoch 8/20: Train Loss: 2.0522, Mutation Acc: 0.4752, Cancer Acc: 0.5104, Val Loss: 2.0513, Val Mutation Acc: 0.4683, Val Cancer Acc: 0.5137
Epoch 9/20: Train Loss: 2.0509, Mutation Acc: 0.4746, Cancer Acc: 0.5098, Val Loss: 2.0512, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5138
Epoch 10/20: Train Loss: 2.0498, Mutation Acc: 0.4755, Cancer Acc: 0.5107, Val Loss: 2.0469, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5141
Epoch 11/20: Train Loss: 2.0501, Mutation Acc: 0.4746, Cancer Acc: 0.5102, Val Loss: 2.0532, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 12/20: Train Loss: 2.0495, Mutation Acc: 0.4751, Cancer Acc: 0.5101, Val Loss: 2.0482, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 13/20: Train Loss: 2.0492, Mutation Acc: 0.4746, Cancer Acc: 0.5104, Val Loss: 2.0472, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 14/20: Train Loss: 2.0486, Mutation Acc: 0.4747, Cancer Acc: 0.5108, Val Loss: 2.0505, Val Mutation Acc: 0.4680, Val Cancer Acc: 0.5142
Epoch 15/20: Train Loss: 2.0490, Mutation Acc: 0.4754, Cancer Acc: 0.5105, Val Loss: 2.0483, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 16/20: Train Loss: 2.0491, Mutation Acc: 0.4747, Cancer Acc: 0.5110, Val Loss: 2.0468, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 17/20: Train Loss: 2.0452, Mutation Acc: 0.4762, Cancer Acc: 0.5114, Val Loss: 2.0467, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 18/20: Train Loss: 2.0450, Mutation Acc: 0.4760, Cancer Acc: 0.5115, Val Loss: 2.0481, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 19/20: Train Loss: 2.0443, Mutation Acc: 0.4756, Cancer Acc: 0.5113, Val Loss: 2.0461, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5142
Epoch 20/20: Train Loss: 2.0438, Mutation Acc: 0.4759, Cancer Acc: 0.5113, Val Loss: 2.0462, Val Mutation Acc: 0.4685, Val Cancer Acc: 0.5143
Final model and encoders saved.