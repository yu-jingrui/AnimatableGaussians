# Train Models Using the MVHumanNet++ Dataset

## MVHumanNet++ Dataset
Download MVHumanNet++ Dataset from the [official repo](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet_plusplus).

MVHumanNet++ is very nice for training AnimatableGaussians models. It features:
- a large number of subjects and outfits 
- 16 cameras surrounding the subject
- 60 frames for each subject-outfit-camera and **an extra A-pose frame**
- camera calibrations, smplx-parameters are accurate and complete

## Preporcessing
If you want to preprocess one single subject, run
```bash
python preprocess_mvhnpp.py --mvhn_subj_path ../../mvhumannet_plusplus/test/100990
```

You can put multiple subjects in a folder (e.g. `../../mvhumannet_plusplus/test/`) and preprocess them together by running:
```bash
python preprocess_mvhnpp.py --mvhn_root ../../mvhumannet_plusplus/test
```

The script will create `calibrations.json` and `smpl_params.npz` in the subject folder. These files conform to the THuman4 format, as the authors of AnimatableGaussians recommended.

## Training

You can follow [GEN_DATA.md](./GEN_DATA.md) and [README.md](../README.md) to train or directly use the script. From the root of AnimatableGaussians
```bash
chmod +x train_mvhnpp.sh
./train_mvhnpp.sh mvhn_100990
```
You must have `template.yaml` and `avatar.yaml` in `./configs/mvhn_100990` for this to work, which we have provided as an example.

### P.S.
I could not configure the env using the given `requirements.txt`. I am using
- `python=3.10`
- `cuda=11.7.1`
- `pytorch-cuda=2.0.1`
- `numpy=1.26.4`
- `pytorch3d=0.7.4` (installed using downloaded pre-compiled file)
- **`scikit-learn=1.1.3`** (different from `requirements.txt`)
- **`scikit-image=0.25.2`** (different from `requirements.txt`)
- **`opencv=4.7.0`** (different from `requirements.txt`)
- **`setuptools=78.1.0`** (different from `requirements.txt`)
- `smplx=0.1.28`