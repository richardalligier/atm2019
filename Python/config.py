import configparser
import os

def read(filename):
    with open(filename) as f:
        sfile = f.read()
    header = "DEFAULT"
    sfile = "[{}]\n".format(header) + sfile
    config = configparser.ConfigParser(delimiters=(':=', '='), inline_comment_prefixes=('#',))
    config.read_string(sfile)
    return dict(config[header])

def resize_figure(f,h_over_w=None):
    size=f.get_size_inches()
    if h_over_w is None:
        h_over_w=size[1]/size[0]
    f.set_tight_layout({'pad':0})
    f.set_figwidth(TEXT_WIDTH)
    f.set_figheight(TEXT_WIDTH*h_over_w)

config = read("config")

DATA_PATH = config['data_path']
GBM_RES = config['gbm_res']
TABLE_FOLDER = config['table_folder']
FIGURE_FOLDER = config['figure_folder']

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
TEXT_WIDTH = 3.6#inch
MODELS = ['DH8D']#['B738','A320','A319','A321','E190','E195','DH8D','B737','CRJ9','A332','B77W']

bmean='${\text{BADA}}_{\text{mean}}$'
bmass='${\text{BADA}}_{\text{mass}}$'
bpred='${\text{BADA}}_{\text{pred}}$'
bpredto='${\text{BADA}}_{\text{pred-take-off}}$'

def pathbadapredicted(method, variables = ""):
        return os.path.join(DATA_PATH, "badapredicted", "_".join([TEST,method,"{}",variables]) + ".csv.xz")

mpath={\
bmean:pathbadapredicted("meanmass","abdemopst"),\
bmass:pathbadapredicted("neuralenergyok","abdemopst"),\
bpred:pathbadapredicted("pred","abdemopst"),\
bpredto:pathbadapredicted("pred","admost"),\
}

vargroupfloat={'b':["massPast","tempkalman","ukalman","vkalman","heading",
           "baroaltitudekalman","velocity","taskalman","vertratecorr"]+\
          ["baroaltitudekalmanm"+str(i) for i in range(1,10)]+["taskalmanm"+str(i) for i in range(1,10)]+\
          ["vertratecorrm"+str(i) for i in range(1,10)]+["temp"+str(i)+"kalman" for i in range(1000,12000,1000)],
               'p':['lat','lon'],
                'e':['energyratem'+str(i) for i in range(1,10)],
                'd':['trip_distance'],
                's':['temp_surfacekalman'],
               'z':['mseEnergyRatePast'],
               }

vargroupcat={
               't':['dayofweek'],
                'i':["icao24"],
                'm':["modeltype"],
                'o':["operator"],
                'a':["fromICAO","toICAO"],
                'c':["callsign"],
               }


def choosevar(vargroup,x):
	return [ x for l in x if l in vargroup for x in vargroup[l]]
