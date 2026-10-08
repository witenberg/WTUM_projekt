# WTUM course project

University project for "Wdrażanie technik uczenia maszynowego" (WTUM), semester 7.
Goal: train an image-classification ML model and deploy it on a smartphone.

Status (2026-10-04): topic chosen, HaGRIDv2 annotations downloaded, EDA done (`notebooks/02_eda.ipynb`),
keypoint path explored (`notebooks/03_path_keypoints.ipynb`, MLP on landmarks 0.993 val macro F1).
No task split, we do everything ourselves. Experiment log: `experiments/log.csv`. Layout and reproduce steps: `README.md`.

Data access: the work Mac's web filter (Mosyle) blocks the HaGRID server (`*.sbercloud.ru`) since 2026-10-04.
Images must be downloaded on another network or machine. Do not try to bypass the filter.

## Topic

Real-time finger counting. The phone camera sees a hand, the app shows how many fingers are extended right now.

Dataset: HaGRID family (hand gesture images). Leaning towards HaGRIDv2, not final yet.
Facts checked on github.com/hukenovs/hagrid on 2026-10-04:

| | HaGRID v1 | HaGRIDv2 |
|---|---|---|
| Images | 552,992 FullHD | 1,086,158 FullHD |
| Gesture classes | 18 | 33 + no_gesture |
| Unique people | 34,730 | 65,977 |
| Full size | 716 GB | 1.5 TB |
| Downsampled (512px) | not listed | 119.4 GB |
| Split | 92/8 train/test | 76/9/15 train/val/test |

Both: bboxes (COCO format), MediaPipe hand landmarks, metadata (age, gender, race). License CC BY-SA 4.0 variant.
Neither has a "finger count" label. We map gesture classes to counts ourselves.

Draft class to count mapping (verify against sample images before using):
- 0: fist
- 1: one, point, like, dislike, mute, little_finger, middle_finger
- 2: peace, peace_inverted, two_up, two_up_inverted, rock, call, thumb_index, thumb_index2
- 3: three, three2, three3, ok, three_gun
- 4: four
- 5: palm, stop, stop_inverted
- Probably drop two-hand and ambiguous ones (hand_heart, hand_heart2, holy, timeout, xsign, take_picture, grabbing, grip). no_gesture could become a "no hand / none" class.
- Class 4 has only one source gesture. Expect imbalance, EDA must show it.

Likely approach (not decided): first reduce the hand to keypoints (e.g. MediaPipe 21 landmarks), then classify the keypoints.
Open question for later: the course diagram expects image preprocessing, image augmentation and xAI like Grad-CAM.
A pure keypoint pipeline changes what those steps mean, so check it covers every requirement.

Target grade: 5.0. Every decision must keep all requirements below satisfied, especially live inference.

## Language

Polish everywhere: chat, notebook text, code comments and docstrings, plot titles and labels, README, docs.
Only code itself (identifiers, file names) is English. This overrides the global "external text in English" rule.

## Decisions

- Platform: Android smartphone.
- App: as simple as possible. One button opens the camera view, the model classifies frames in real time with the result overlaid.
  No extra screens or features. All effort goes into the ML process and its documentation.
- Model must be small and fast enough for live on-device inference (pick architecture and export format with that in mind).

## Deliverables

All three are required:
1. Project (code, model, mobile app)
2. Presentation (PDF uploaded to Teams, folder `prezentacja`, before the class)
3. Documentation (PDF uploaded to Teams, folder `dokumentacja`, before the class)

## Schedule (semester weeks)

- Week 1: introduction
- Weeks 2-6: working sessions and consultations in room B318
- Week 7: presentations
- Week 8: final assessment combined with the lecture + documentation

Attendance required at meetings 1, 7, 8 plus at least 3 consultation sessions.
Consultations of other groups count too.

## Technical requirements

Data source: image files.

- EDA: class balance, number of classes, data quality
- Preprocessing: how did it change the results?
- Augmentation: how did it change the results?
- Hyperparameter tuning: how did it change the results?
- GreenAI metrics (energy, emissions, training time etc.)
- Explainability (xAI): SHAP, LIME, Grad-CAM or other
- Deployment to a smartphone
- Verification with a video recording

Grading trap: a video that is not analysed in near real time caps the grade at 4.0.
So the phone must run inference live on camera, not on pre-recorded footage processed offline.

## Required process

Diagram: `docs/proces_projektowy.png`. Documentation and presentation must follow it.

1. Choose a topic based on a review of available datasets
2. Baseline model (CNN, classifier, tree-based model, ...)
3. Image preprocessing
4. Gate: did results AND GreenAI metrics improve? If not, go back to the previous model.
5. Augmentation
6. Gate: same check, otherwise revert
7. Hyperparameter tuning
8. Gate: same check, otherwise revert
9. Final model
10. xAI
11. Deployment
12. Mobile app tests

Every step needs before/after numbers for both quality metrics and GreenAI metrics.
Keep a log of each experiment so the gates can be shown in the docs.

## Documentation rules

- Focus on the process (the diagram above)
- Photos and diagrams welcome
- GreenAI can be compared to something tangible, e.g. average car fuel use over a distance
- Show xAI on concrete examples
- A notebook can be the base, but it must be "dressed" with extra descriptions
- Conclusions: what we learned, what we take away, what was hard, which tech we used for the first time
- Wrong design paths are welcome, describe why they were wrong. So log failed attempts too.
- Words matter. Evaluation != augmentation != normalization etc. Use terms precisely.
- Export to PDF and upload to Teams (`dokumentacja`) before the class

## Presentation rules

- 10 minutes, watch the time. Treat it as a rehearsal for the thesis defence.
- Use the WTIiE template
- Focus on the process (the diagram)
- Answer questions from the audience. The quality of questions we ask others is graded too.
- Export to PDF and upload to Teams (`prezentacja`) before the class

## Original instructor messages (Polish, verbatim)

### Sprawy organizacyjne

Szanowni Państwo, zgodnie z wymaganiami regulaminu przedstawiam zasady zaliczenia zajęć projektowych z WTUM.

Harmonogram pracy (w podziale na tygodnie semestru):
1. wprowadzenie
2 - 6. robocze, konsultacyjne w B318
7. prezentacje
8. zaliczenie łączone z wykładem + dokumentacja

Obecności: spotkania 1, 7, 8 + min. 3 wybrane terminy zajęć (konsultacje) - można przychodzić również na terminy konsultacyjne innych grup!

Będziecie Państwo realizować projekt dot. uczenia maszynowego oraz jego implementacji na urządzeniach mobilnych. Jako źródło danych wykorzystane zostaną pliki graficzne. Wymagania projektowe:
- niezbędne jest dostarczenie trzech części - projekt + prezentacja + dokumentacja
- eksploracyjna analiza danych - balans? liczba klas? jakość danych?
- przetwarzanie wstępne - jak pre-processing wpłynął na wyniki?
- augmentacja - jak augmentacja wpłynęła na wyniki?
- strojenie parametrów - jak wpłynęło na wyniki?
- metryki GreenAI
- wyjaśnialność (SHAP, Lime, GradCAM, inne)
- wdrożenie --> smartfon
- weryfikacja FILMEM!

Weryfikacja nagranym filmem, ale nie przeanalizowanym w czasie zbliżonym do rzeczywistego, obniża ocenę do max. 4.0.

### Proces projektowy (diagram)

wybór tematu na podstawie przeglądu dostępnych zbiorów danych → model baseline (model CNN, klasyfikator, struktura drzewiasta, ..) → przetwarzanie wstępne obrazów → [czy wyniki i wskaźniki greenAI się poprawiły? nie: wróć do poprzedniego modelu] → augmentacja → [ta sama bramka] → strojenie parametrów → [ta sama bramka] → ostateczny model → xAI → wdrożenie → testy aplikacji mobilnej

### Dokumentacja

- Skupiamy się na procesie (patrz schemat)
- Zdjęcia / schematy mile widziane
- GreenAI można przedstawić w porównaniu np. do średniego zużycia samochodu na dystansie
- xAI pokazujemy na przykładzie
- Wyraźny podział zadań wewnątrz grupy projektowej
- Bazą do dokumentacji może być notebook, ale trzeba go "ubrać" w dodatkowe opisy
- WNIOSKI – czego się nauczyliśmy? co "zabieramy ze sobą" po projekcie? co było wyzwaniem? z której technologii korzystaliśmy po raz pierwszy?
- Błędne ścieżki projektowe mile widziane – opisujemy dlaczego były błędne
- Słowa mają znaczenie (ewaluacja != augmentacja != normalizacja itp.) - przeczytajmy tę dokumentację!
- Wyeksportowaną przed zajęciami wrzucamy w PDF na Teams (folder: dokumentacja)

### Prezentacja

- zwróćmy uwagę na czas (10 minut) i warsztat prezentującego - można potraktować to jak próbę przed obroną!
- szablon WTIiE
- skupiamy się na PROCESIE (patrz schemat)
- każdy z członków grupy opowiada o swojej części projektu
- odpowiadamy na pytania z sali (jakość pytań również jest oceniana)
- wyeksportowaną przed zajęciami wrzucamy w PDF na Teams (folder: prezentacja)
