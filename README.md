# Chest X-ray Pathology Detection (YOLOv8 + Faster R-CNN)

Projet Traitement d'image — M2 IASD. Détection de 8 pathologies thoraciques
(Atelectasis, Cardiomegaly, Effusion, Infiltrate, Mass, Nodule, Pneumonia,
Pneumothorax) sur radiographies 1024×1024 (sous-ensemble NIH ChestX-ray14).
Métrique : mAP @ IoU 0.5.

## Structure

```
input/m-2-big-data-dlia-project-2025-clean-dataset/
    train/  (710 images)   train.csv (814 boîtes)
    test/   (170 images)   ID_to_Image_Mapping.csv
notebooks/
    01_yolov8_fasterrcnn_ensemble.ipynb      YOLOv8-L + Faster R-CNN (ancres par défaut) + fusion NMS
    02_fasterrcnn_clahe_anchors_final.ipynb  version finale : CLAHE, petites ancres, AdamW, mAP de validation
models/
    fasterrcnn_chestxray.pth   Faster R-CNN ResNet50-FPN, 9 classes, ANCRES PAR DÉFAUT (notebook 01)
submissions/
    submission.csv        YOLOv8 + TTA + WBF
    submission_frcnn.csv  Faster R-CNN seul
    submission_both.csv   ensemble YOLOv8 + Faster R-CNN
presentation/
    Chest_Xray_Detection_Presentation.pptx
archive/
    draft_unexecuted.ipynb    brouillon non exécuté (= notebook 01 sans la ligne `device`)
```

Le dossier s'appelle `input/` exprès : les notebooks utilisent les chemins Kaggle
`../input/m-2-big-data-dlia-project-2025-clean-dataset/...`, qui fonctionnent donc
tels quels quand on lance les notebooks depuis `notebooks/`.

## Vérification (06/10/2026)

- Données : 710 images train = 710 ids dans train.csv, 170 images test = 170 ids
  du mapping, aucun fichier manquant, aucune boîte invalide, pas de fuite train/test.
- Submissions : les 3 CSV ont 170 lignes, un ID unique par image, IDs conformes
  au mapping, confiances dans [0,1], boîtes valides.
- Modèle : `fasterrcnn_chestxray.pth` se charge sans erreur dans
  `fasterrcnn_resnet50_fpn` (9 classes, ancres par défaut) et produit des prédictions.

## Points d'attention

- `fasterrcnn_chestxray.pth` n'est PAS le modèle de `submission_frcnn.csv` : ses
  prédictions diffèrent (ex. 00000150_002 → Cardiomegaly vs Nodule dans le CSV).
  Il ne se charge pas non plus avec la cellule de chargement du notebook 02 (ancres
  personnalisées, 6 ancres/position vs 3). Le checkpoint final `best_frcnn_cxr.pth`
  et les poids YOLO `runs/detect/train/weights/best.pt` sont restés sur Kaggle.
- Les notebooks sauvegardent `fasterrcnn_best.pth` / `best_frcnn_cxr.pth`, pas
  `fasterrcnn_chestxray.pth`.
- Score de validation du notebook 02 faible : meilleur mAP 0.036 (mAP50 ≈ 0.08,
  epoch 10), puis sur-apprentissage (val_loss augmente).
- Entraînement prévu pour GPU (Kaggle P100, `device=0`) ; YOLOv8-L à 1024 px
  n'est pas réaliste sur CPU.

## Lancer

```
pip install ultralytics ensemble-boxes torch torchvision scikit-learn pandas pillow tqdm
cd notebooks && jupyter notebook
```
