# Predictive Distribution of the Mass and Speed Profile to Improve Aircraft Climb Prediction

This Github page hosts the code producing the results published in the paper "Predictive Distribution of the Mass and Speed Profile to Improve Aircraft Climb Prediction".

The trajectory data are automatically downloaded by the script. They are hosted at [https://opensky-network.org/datasets/publication-data/climbing-aircraft-dataset](https://opensky-network.org/datasets/publication-data/climbing-aircraft-dataset).

With this code, you can reproduce the Tables 3, 4, 5 and Figures 5, 6 of the publication. In order to reproduce the Tables, you must have computed the prediction with the GBM method using the repository of a previous publication [https://github.com/richardalligier/trc2018](https://github.com/richardalligier/trc2018).

If you have any problems using the provided code, please feel free to open an issue in this Github repository.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run the Python3 scripts, you will need to install different packages. These packages can be installed with the command:

```
pip3 install pandas numpy matplotlib seaborn scikit-learn scipy torch==1.0.1
```

In order to compile the OCaml binaries, you will need to install the OCaml compiler. Using Debian/Ubuntu, just type:

```
apt-get update
apt-get install ocaml ocaml-native-compilers
```

### Installing
To install the project, you just have to clone or download this github repository. To clone this repository, just type:

```
git clone https://github.com/richardalligier/atm2019.git
```
## Running the Scripts

### Configuring the Scripts

Before running the scripts you might want to edit the file `config`. In this file, you can edit where the generated tables and figures will be created by modifying the variables `FIGURE_FOLDER` and `TABLE_FOLDER`. Likewise, you can edit `DATA_PATH`, this variable is the folder storing the trajectory data, the generated models and predictions. The trajectory data are automatically downloaded.

If you want to reproduce the Tables, you must have computed the prediction with the GBM method using the repository of a previous publication [https://github.com/richardalligier/trc2018](https://github.com/richardalligier/trc2018). The `GBM_RES` variable in the `config` will specify the data folder used to reproduce the TRC2018 results.


### Computing the Predicted Distribution of the Mass and Speed

As a reminder, the predicted distribution is specific, tailored, to each point of each flight. To test the script, you might want to compute it considering only the flight of a given aircraft type. For instance, if you want to compute for the DH8D flights, just type:

```
make MODELS="DH8D"
```

To compute the predicted distributions for all the aircraft types, you only have to type (**WARNING: Takes a lot of time!!**) :


```
make
```

This script uses 4 thread and the GPU. It takes several days (maybe a week depending on your computer). Depending on the aircraft type being computed it can take up to approximately 9GB of RAM.


### Reproducing Figures

Figures 5, 6 can be computed without computing the operational factors. For the other tables and figures, the predicted operational factors must have been computed.

You can use the command `make` to compute the figures, just type:

```
make figures
```

If you only want some, remove the figures or tables you do not want.


### Reproducing Tables

Assuming that the folder `GBM_RES` contains the GBM predicted values using the publication [https://github.com/richardalligier/trc2018](https://github.com/richardalligier/trc2018), you just have to type:

```
make tables
```

## Author

* **Richard Alligier**

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details

## Acknowledgments
* Jingwei Zhang ([@jingweiz](https://github.com/jingweiz)) author of the adamw.py file for allowing me host this file in this repository
* [The OpenSky Network](https://opensky-network.org/) for providing and hosting the trajectory data
* [FlightAirMap](https://www.flightairmap.com/) for providing data on Routes and ICAO codes
* [World Aircraft Database](https://junzisun.com/adb/) for providing data on ICAO codes
* NVIDIA Corporation with the donation of the Titan Xp GPU used for this research

## Appendix: Data Description

The data description is available [here](https://github.com/richardalligier/trc2018/blob/master/README.md#appendix-data-description).
