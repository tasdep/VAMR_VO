## About The Project
Our VAMR VO pipeline project :)

## Getting Started
To get a local copy up and running follow these steps.

### Prerequisites
1) Install [conda](https://www.anaconda.com/download)
2) Clone this repo

### Installation
1) Navigate to the project directory and run `conda env create -f environment.yml`
2) Run `conda activate vamr_vo`

### Running
1) Navigate to the VAMR_VO directory
2) Run `python visual_odometry/main.py`

## Links
- [Subject website](https://rpg.ifi.uzh.ch/teaching.html)
- [Project FAQ website](https://docs.google.com/document/d/1IuWmXyO1m5DV77AhEa-MpK-yp9LDMQO5IM6oyVLvHa0/edit#heading=h.w8vo6xo5cuee)
- [Overleaf draft report](https://www.overleaf.com/1966191675jhdpcgnggbfs#a129d1)
- [4.3 Pseudocode](https://docs.google.com/document/d/1G66pdkz_V2rK24olrELUtHSQnHHMSyiDgMahU2Wfno0/edit)

## Datasets
The shared data folder contains the small/easy dataset. The other larger datasets should be stored locally in a top level folder `local_data/`. This is in the `.gitignore`.

## Development Guidelines
- Do work in your own branch, with the prefix of your github id followed by feature name. Ex: `git checkout -b 'kappibw/project-skeleton'`
- Annotate all np arrays with the expected shape/dimensions in inline comments.
- Add someone else as reviewer on a PR before merging to master.
- Document your work in the draft report as you go, use it as a scratchbook.

