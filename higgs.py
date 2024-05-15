import uproot
import hist
from hist import Hist
from TLorentzVector import TLorentzVector
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

#plt.rcParams['text.usetex'] = True

def mcWeights(data,norm,lumi=10):
    """
    When MC simulation is compared to data the contribution of each simulated event needs to be
    scaled ('reweighted') to account for differences in how some objects behave in simulation
    vs in data, as well as the fact that there are different numbers of events in the MC tree than 
    in the data tree.
    
    Parameters
    ----------
    tree : TTree entry for this event
    """
    
    scaleFactor_ELE = data["scaleFactor_ELE"]
    scaleFactor_MUON = data["scaleFactor_MUON"]
    scaleFactor_LepTRIGGER = data["scaleFactor_LepTRIGGER"]
    scaleFactor_PILEUP = data["scaleFactor_PILEUP"]
    mcWeight = data["mcWeight"]
    #These values do change from event to event
    scale_factors = scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER*scaleFactor_PILEUP*mcWeight
    
    weight = norm*scale_factors
    return weight

requiredData = ["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type",
                "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM",
                "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER",
                "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi", "jet_pt", "jet_n", "jet_MV2c10", "runNumber"]

# cuts

# leptons are well isolated
# 0 or 1 jets, 0 bjets
# exactly two good leptons
# different flavour leptons
# opposite charge of leptons
# lepA pt > 22000, lepB pt > 15000
# lepton separation < 1.8 phi
# dilep pt > 30000
# 55000 > dilep M > 10000
# MET pt > 30000
# MET-dilepton separation > 1.2 phi

def hWW(data,hist,mc, mul=1):
    """
    Function which executes the analysis flow for the Higgs production cross-section measurement in the H->WW
    decay channel.
    
    Fills a histogram with mT(llvv) of events which pass the full set of cuts 
    
    Parameters
    ----------
    data : A Ttree containing data / background information
    
    hist : The name of the histogram to be filled with mT(llvv) values
    
    mode : A flag to tell the function if it is looping over 'data' or 'mc'
    """
    
    XSection = data["XSection"]
    SumWeights = data["SumWeights"]
    #These values don't change from event to event
    norm = 10*(XSection[0]*1000)/SumWeights[0]
    
    startTime = perf_counter()
    print(f"pre-filtering {len(data)} data elements")
    # needs to pass either the electron or muon trigger
    data = data[data["trigE"] + data["trigM"]]
    #data = data[data["lep_isTightID"]]
    print(f"filtered in {perf_counter() - startTime}s, there is now {len(data)} elements.")
    
    startTime = perf_counter()
    for n, event in enumerate(data):
        
        #############################
        ### Event-level requirements
        #############################
    
        if n & 0xFFFF == 0:
            print(f"{n} iterations passed")
            
        ####Lepton preselections
        #Initialise (set up) the variables we want to return
        goodLeps = [] #Indices (position in list of event's leptons) of our good leptons
        lep_n = event["lep_n"]
        ##Loop through all the leptons in the event
        for j in range(0,lep_n):
            lep_isTightID = event["lep_isTightID"][j]    
            ##Check lepton ID
            if(lep_isTightID):
                lep_ptcone30 = event["lep_ptcone30"][j]
                lep_pt = event["lep_pt"][j]
                lep_etcone20 = event["lep_etcone20"][j]
                #Check lepton isolation
                #Similar to photonIsolation() above, different thresholds
                if lep_ptcone30 / lep_pt < 0.1 and \
                   lep_etcone20 / lep_pt < 0.1:

                    #Only central leptons 
                    #Electrons and muons have slightly different eta requirements
                    lep_type = event["lep_type"][j]
                    lep_eta = event["lep_eta"][j]
                    #Electrons: 'Particle type code' = 11
                    if lep_type == 11:
                        #Check lepton eta is in the 'central' region and not in "transition region" 
                        if (np.abs(lep_eta) < 2.37) and\
                           (np.abs(lep_eta) < 1.37 or np.abs(lep_eta) > 1.52): 

                            goodLeps.append(j) #Store lepton's index

                    #Muons: 'Particle type code' = 13
                    elif (lep_type == 13) and (np.abs(lep_eta) < 2.5): #Check'central' region

                            goodLeps.append(j) #Store lepton's index
                            
        # elem if bad jets
        jet_n = event["jet_n"]
        pt = event["jet_pt"]
        p, q = 0, 0
        for j in range(0, jet_n):
            if event["jet_MV2c10"][j] > 0.18 and pt[j] > 20000:
                p += 1
            elif pt[j] > 30000:
                q += 1
        if p != 0 or q >= 2:
            continue

        ###################################
        ### Individual lepton requirements
        ###################################

        if len(goodLeps) != 2: #Exactly two good leptons...
            continue
            
        lep1 = goodLeps[0] #INDICES of the good leptons
        lep2 = goodLeps[1]

        lep_type = event["lep_type"]
        if lep_type[lep1] == lep_type[lep2]: #... with different flavour
            continue

        lep_charge = event["lep_charge"]
        if lep_charge[lep1] == lep_charge[lep2]: #... and opposite charge...
            continue

        lep_pt = event["lep_pt"]
        if lep_pt[lep1] > 22000 and lep_pt[lep2] > 15000: #pT requirements
            #Note: TTrees always sort objects in descending pT order

            lep_phi = event["lep_phi"]
            if abs(lep_phi[lep1] - lep_phi[lep2]) < 1.8: #lepton separtion in phi # was 1.8

                #################################
                ### Dilepton system requirements
                #################################

                #Initialse (set up) an empty 4 vector for dilepton system
                dilep_four_mmtm = TLorentzVector()

                #Loop through our list of lepton indices
                for i in goodLeps:

                    #Initialse (set up) an empty 4 vector for each lepton
                    lep_i = TLorentzVector()

                    lep_pt = event["lep_pt"][i]
                    lep_eta = event["lep_eta"][i]
                    lep_phi = event["lep_phi"][i]
                    lep_E = event["lep_E"][i]
                    #Retrieve the lepton's 4 momentum components from the tree
                    lep_i.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)

                    #Store lepton's 4 momentum
                    dilep_four_mmtm += lep_i

                # Dilepton system pT > 30 GeV
                if dilep_four_mmtm.Pt() > 30000:

                    if dilep_four_mmtm.M() > 10000 and dilep_four_mmtm.M() < 55000:

                        #####################
                        ### MET requirements
                        #####################

                        met_et = event["met_et"]
                        met_phi = event["met_phi"]
                        #Initialse (set up) an empty 4 vector for the event's MET and fill from tree
                        met_four_mom = TLorentzVector()
                        met_four_mom.SetPtEtaPhiE(met_et, 0, met_phi, met_et)

                        #MET > 30 GeV
                        if met_four_mom.Pt() > 30000:

                            #Diffence in phi between the dilepton system and the MET < pi/2 (1.57)
                            if abs(dilep_four_mmtm.Phi()-met_four_mom.Phi()) > 1.2:

                                #####################
                                ### Full llvv system
                                #####################
                                system_four_mom = dilep_four_mmtm + met_four_mom

                                #Use the keyword weight to specify the weight of the evwnt
                                #If event is MC: Reweight it
                                if mc: weight = mcWeights(event, norm)
                                else: weight = 1

                                hist.fill(system_four_mom.Mt()/1000, weight=weight*mul)
    print(f"finished processing in {perf_counter() - startTime}s")

def analyseRealData():
    h_dat = Hist(hist.axis.Regular(30, 60, 300, label = "Transverse mass m_{T}"))

    # real = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/Data/data_A.2lep.root")
    # dataTree = real["mini"]
    # numDataEntries = len(dataTree["runNumber"].array())
    # print("Tree contains", numDataEntries, "entries")
    # data = dataTree.arrays(requiredData)
    # fNumerator = 1
    # fractionOfEvents = int(numDataEntries/fNumerator)
    # hWW(data[:fractionOfEvents], h_dat, False)

    for i, c in enumerate(("A", "B", "C", "D")):
        real = uproot.open(f"https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/Data/data_{c}.2lep.root")
        dataTree = real["mini"]
        numDataEntries = len(dataTree["runNumber"].array())
        print("Tree contains", numDataEntries, "entries")

        data = dataTree.arrays(requiredData)

        fNumerator = 1
        fractionOfEvents = int(numDataEntries/fNumerator)
        h_dat = Hist(hist.axis.Regular(30, 60, 300, label = "Transverse mass m_{T}"))
        hWW(data[:fractionOfEvents], h_dat, False)
        
        toSave = [[round(k) for k in h_dat.to_numpy()[1].tolist()], h_dat.values().tolist()]
        with open(f"save{c}.csv", "w") as f:
            for v in zip(*toSave):
                f.write(",".join(str(n) for n in v) + "\n")

    #h_dat.plot(histtype = "fill")
    #plt.show()

def analyseBackgroundData():
    backgroundPaths = [
        #"2lep/MC/mc_363492.llvv.2lep",               # WW
        
        #"2lep/MC/mc_410000.ttbar_lep.2lep"           # ttbar
        
        #"mc_410011.single_top_tchan.2lep.root",      # single t
        
        #"mc_410012.single_antitop_tchan.2lep.root",
        #"mc_410013.single_top_wtchan.2lep.root",
        #"mc_410014.single_antitop_wtchan.2lep.root",
        #"mc_410025.single_top_schan.2lep.root",
        #"mc_410026.single_antitop_schan.2lep.root"
        
        #"mc_361100.Wplusenu.2lep.root",              # W+jets
        #"mc_361101.Wplusmunu.2lep.root",
        #"mc_361102.Wplustaunu.2lep.root",
        #"mc_361103.Wminusenu.2lep.root",
        #"mc_361104.Wminusmunu.2lep.root",
        #"mc_361105.Wminustaunu.2lep.root"
        
        "mc_364158.Wmunu_PTV0_70_BFilter.2lep.root",  # W+jets but worse apparently (not included)
        "mc_364161.Wmunu_PTV70_140_BFilter.2lep.root",
        "mc_364164.Wmunu_PTV140_280_BFilter.2lep.root",
        "mc_364167.Wmunu_PTV280_500_BFilter.2lep.root",
        "mc_364168.Wmunu_PTV500_1000.2lep.root",
        "mc_364169.Wmunu_PTV1000_E_CMS.2lep.root",
        
        "mc_364172.Wenu_PTV0_70_BFilter.2lep.root",
        "mc_364175.Wenu_PTV70_140_BFilter.2lep.root",
        "mc_364178.Wenu_PTV140_280_BFilter.2lep.root",
        "mc_364181.Wenu_PTV280_500_BFilter.2lep.root",
        "mc_364182.Wenu_PTV500_1000.2lep.root",
        "mc_364183.Wenu_PTV1000_E_CMS.2lep.root"
    ]

    backgrounds = [uproot.open(f"https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/MC/{path}:mini") for path in backgroundPaths]
    mcSim = [bg.arrays(requiredData) for bg in backgrounds]
    bgplots = [Hist(hist.axis.Regular(30, 60, 300, label = "Transverse mass m_{T}")) for _ in backgrounds]
    #print("Tree contains", numMCEntries, "entries") 

    for bg in range(len(backgrounds)):
        fNumerator = 1
        numMCEntries = len(backgrounds[bg]["runNumber"].array())
        print(numMCEntries)
        fractionOfMC = int(numMCEntries/fNumerator)
        hWW(mcSim[bg][0:fractionOfMC],bgplots[bg],True,1)
        
        toSave = [[round(k) for k in bgplots[bg].to_numpy()[1].tolist()], bgplots[bg].values().tolist()]
        with open(f"saveBGwJetsAgain{bg}.csv", "w") as f:
            for v in zip(*toSave):
                f.write(",".join(str(n) for n in v) + "\n")

    for bgplot in bgplots:
        bgplot.plot(histtype = "fill")
        plt.show()

# toSave = [[round(k) for k in h_dat.to_numpy()[1].tolist()]] + [k.values().tolist() for k in (h_dat, *bgplots)]
# with open("save.csv", "w") as f:
#     for v in zip(*toSave):
#         f.write(",".join(str(n) for n in v) + "\n")

def displayPlots():
    # with open("save.csv", "r") as f:
    #     dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
    #     #print(dat)
 
    # for j in range(1, 4):
    #     h = Hist(hist.axis.Regular(30, 60, 300, label = f"{j}"))
    #     for i, v in zip(dat[0],dat[j]):
    #         h.fill(i, weight=v)
    #     h.plot(histtype="fill")
    # plt.show()

    h = Hist(hist.axis.Regular(30, 60, 300, label = "aaa"))
    for c in "A", "B", "C", "D":
        with open(f"data/save{c}.csv", "r") as f:
            dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
            #print(dat)

            for i, v in zip(dat[0],dat[1]):
                h.fill(i, weight=v)
    #h.plot(histtype="fill")
    #plt.show()

    ha = Hist(hist.axis.Regular(30, 60, 300, label = "aaa"))

    bg = Hist(
        hist.axis.Regular(30, 60, 300, label = "aaa"), 
        hist.axis.StrCategory(["single t", "W + jets", "tt", "WW"], name = "c")
    )


    with open("data/saveBG_llvv.csv", "r") as f:
        dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
        for i, v in zip(dat[0],dat[1]):
            bg.fill(i, weight=v, c="WW")
            ha.fill(i, weight=v)
            
    with open("data/saveBG_ttbar.csv", "r") as f:
        dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
        for i, v in zip(dat[0],dat[1]):
            bg.fill(i, weight=v, c="ttbar")
            ha.fill(i, weight=v)
            
    for n in range(6):
        with open(f"data/saveBGTop{n}.csv", "r") as f:
            dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
            for i, v in zip(dat[0],dat[1]):
                bg.fill(i, weight=v, c="single t")
                ha.fill(i, weight=v)
                
    for n in range(6):
        with open(f"data/saveBGwJets{n}.csv", "r") as f:
            dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
            for i, v in zip(dat[0],dat[1]):
                bg.fill(i, weight=v * 0.3, c="W + jets")
                ha.fill(i, weight=v * 0.3)
            
    #for n in range(12):
    #    with open(f"data/saveBGwJetsAgain{n}.csv", "r") as f:
    #        dat = list(zip(*[[float(v) for v in line.split(",")] for line in f.read().strip().split("\n")]))
    #        for i, v in zip(dat[0],dat[1]):
    #            bg.fill(i, weight=v, c="W + jets")
    #            ha.fill(i, weight=v)

    bg.plot(histtype="fill", stack=True)
    plt.legend()
    plt.show()

    h.plot(histtype="fill")
    #(ha * 1.8).plot(histtype="fill", stack=True)
    #hb.plot(histtype="fill", stack=True)
    (bg * 1.26).plot(histtype="fill", stack=True)
    plt.legend()
    plt.show()

    (h - ha * 1.26).plot(histtype="fill")
    plt.show()

if __name__ == "__main__":
    displayPlots()