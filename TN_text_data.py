import numpy as np
import pandas as pd
import re
import torch


def get_dna_methylation_by_case_id(case_id):
    label_table = pd.read_csv('Step_4.csv', encoding='gbk')
    dna_methylation_27_table = pd.read_csv('./data/DNA_Methylation_27_60gene.csv', encoding='gbk')
    dna_methylation_450_table = pd.read_csv('./data/DNA_Methylation_450_60gene.csv', encoding='gbk')
    label_row = label_table[label_table['case_id']==case_id]
    label = label_row.loc[label_row.index, ['DNA_Methylation']]
    label = label.values
    label = label[0,0]
    if(label == 450):
        dna_methylation_450_row = dna_methylation_450_table[dna_methylation_450_table['case_id']==case_id]
        #20
        """dna_methylation_data = dna_methylation_450_row.loc[dna_methylation_450_row.index, ["cg00298481", "cg25959752", "cg24358246", "cg17869315", "cg10364881", "cg12744858", "cg07363131", "cg11020894", "cg05982389", "cg25102842",
                                                                                           "cg19484680", "cg20884043", "cg16241861", "cg18842310", "cg14118583", "cg00157562", "cg14227125", "cg07380384", "cg25209017", "cg12108555"]]"""
        #40
        """dna_methylation_data = dna_methylation_450_row.loc[
            dna_methylation_450_row.index, ["cg24358246", "cg00298481", "cg25959752", "cg17869315", "cg10364881",
                                            "cg12744858", "cg07363131", "cg11020894", "cg05982389", "cg25102842",
                                            "cg19484680", "cg20884043", "cg16241861", "cg18842310", "cg14118583",
                                            "cg00157562", "cg14227125", "cg07380384", "cg25209017", "cg12108555",
                                            "cg14587704", "cg04140754", "cg27177808", "cg20119464", "cg13226597",
                                            "cg19712123", "cg21315000", "cg02671826", "cg16672659", "cg19907968",
                                            "cg15279800", "cg15973546", "cg00730348", "cg04788722", "cg22357390",
                                            "cg06829788", "cg26820118", "cg12411965", "cg26545278", "cg16258503",
                                            ]]"""
        #60
        dna_methylation_data = dna_methylation_450_row.loc[dna_methylation_450_row.index, ["cg24358246", "cg00298481", "cg25959752", "cg17869315", "cg10364881", "cg12744858", "cg07363131", "cg11020894", "cg05982389", "cg25102842",
                                                                                              "cg19484680", "cg20884043", "cg16241861", "cg18842310", "cg14118583", "cg00157562", "cg14227125", "cg07380384", "cg25209017", "cg12108555",
                                                                                              "cg14587704", "cg04140754", "cg27177808", "cg20119464", "cg13226597", "cg19712123", "cg21315000", "cg02671826", "cg16672659", "cg19907968",
                                                                                              "cg15279800", "cg15973546", "cg00730348", "cg04788722", "cg22357390", "cg06829788", "cg26820118", "cg12411965", "cg26545278", "cg16258503",
                                                                                              "cg14062643", "cg26521129", "cg03612577", "cg20988215", "cg03722068", "cg20024324", "cg21853948", "cg20902566", "cg05881745", "cg24480515",
                                                                                              "cg10581256", "cg02862897", "cg18279842", "cg06625244", "cg00473769", "cg23043143", "cg00971804", "cg11204780", "cg15447511", "cg07122319"]]
        #80
        """dna_methylation_data = dna_methylation_450_row.loc[
            dna_methylation_450_row.index, ["cg24358246", "cg00298481", "cg25959752", "cg17869315", "cg10364881",
                                            "cg12744858", "cg07363131", "cg11020894", "cg05982389", "cg25102842",
                                            "cg19484680", "cg20884043", "cg16241861", "cg18842310", "cg14118583",
                                            "cg00157562", "cg14227125", "cg07380384", "cg25209017", "cg12108555",
                                            "cg14587704", "cg04140754", "cg27177808", "cg20119464", "cg13226597",
                                            "cg19712123", "cg21315000", "cg02671826", "cg16672659", "cg19907968",
                                            "cg15279800", "cg15973546", "cg00730348", "cg04788722", "cg22357390",
                                            "cg06829788", "cg26820118", "cg12411965", "cg26545278", "cg16258503",
                                            "cg14062643", "cg26521129", "cg03612577", "cg20988215", "cg03722068",
                                            "cg20024324", "cg21853948", "cg20902566", "cg05881745", "cg24480515",
                                            "cg10581256", "cg02862897", "cg18279842", "cg06625244", "cg00473769",
                                            "cg23043143", "cg00971804", "cg11204780", "cg15447511", "cg07122319",
                                            "cg14302083", "cg18097327", "cg06740049", "cg05781820", "cg20212362",
                                            "cg05641048", "cg06459724", "cg24365867", "cg02265740", "cg15111398",
                                            "cg24391989", "cg26119731", "cg13200354", "cg22128468", "cg06509140",
                                            "cg23531640", "cg27378762", "cg07128873", "cg12202228", "cg18117895"]]"""


    else:
        dna_methylation_27_row =dna_methylation_27_table[dna_methylation_27_table['case_id']==case_id]
        #20
        """dna_methylation_data = dna_methylation_27_row.loc[
        dna_methylation_27_row.index, ["cg22153873", "cg11564670", "cg26900154", "cg25775449", "cg10636246", "cg24076884", "cg00033773", "cg27195917", "cg14310890", "cg18847227",
                                       "cg09088577", "cg09395833", "cg13086467", "cg18881723", "cg21006686", "cg07115820", "cg08220793", "cg08359956", "cg26284390", "cg07806164"]]"""
        #40
        """dna_methylation_data = dna_methylation_27_row.loc[
            dna_methylation_27_row.index, ["cg00033773", "cg00627286", "cg00644033", "cg02456292", "cg03835332",
                                           "cg03966406", "cg04128563", "cg04368877", "cg04882894", "cg05208878",
                                           "cg06090864", "cg06134936", "cg06245154", "cg06751597", "cg06839953",
                                           "cg07115820", "cg07535879", "cg07740640", "cg07806164", "cg08220793",
                                           "cg08315187", "cg08359956", "cg08433095", "cg08612601", "cg08744769",
                                           "cg08875535", "cg09088577", "cg09118625", "cg09395833", "cg10636246",
                                           "cg11564670", "cg11748006", "cg13086467", "cg14310890", "cg14481339",
                                           "cg15076218", "cg15423862", "cg17749384", "cg17891149", "cg17942096",
                                           ]]"""
        #60
        dna_methylation_data = dna_methylation_27_row.loc[
                    dna_methylation_27_row.index, ["cg00033773", "cg00627286", "cg00644033", "cg02456292", "cg03835332", "cg03966406", "cg04128563", "cg04368877", "cg04882894", "cg05208878",
                                                   "cg06090864", "cg06134936", "cg06245154", "cg06751597", "cg06839953", "cg07115820", "cg07535879", "cg07740640", "cg07806164", "cg08220793",
                                                   "cg08315187", "cg08359956", "cg08433095", "cg08612601", "cg08744769", "cg08875535", "cg09088577", "cg09118625", "cg09395833", "cg10636246",
                                                   "cg11564670", "cg11748006", "cg13086467", "cg14310890", "cg14481339", "cg15076218", "cg15423862", "cg17749384", "cg17891149", "cg17942096",
                                                   "cg17959722", "cg18070061", "cg18595258", "cg18847227", "cg18881723", "cg19632760", "cg20098887", "cg20311501", "cg21006686", "cg22153873",
                                                   "cg22189286", "cg22578610", "cg24076884", "cg24983959", "cg25775449", "cg26284390", "cg26900154", "cg27039606", "cg27195917", "cg27195917"]]
        #80
        """dna_methylation_data = dna_methylation_27_row.loc[
                    dna_methylation_27_row.index, ["cg00033773", "cg00627286", "cg00644033", "cg02456292", "cg03835332",
                                                   "cg03966406", "cg04128563", "cg04368877", "cg04882894", "cg05208878",
                                                   "cg06090864", "cg06134936", "cg06245154", "cg06751597", "cg06839953",
                                                   "cg07115820", "cg07535879", "cg07740640", "cg07806164", "cg08220793",
                                                   "cg08315187", "cg08359956", "cg08433095", "cg08612601", "cg08744769",
                                                   "cg08875535", "cg09088577", "cg09118625", "cg09395833", "cg10636246",
                                                   "cg11564670", "cg11748006", "cg13086467", "cg14310890", "cg14481339",
                                                   "cg15076218", "cg15423862", "cg17749384", "cg17891149", "cg17942096",
                                                   "cg17959722", "cg18070061", "cg18595258", "cg18847227", "cg18881723",
                                                   "cg19632760", "cg20098887", "cg20311501", "cg21006686", "cg22153873",
                                                   "cg22189286", "cg22578610", "cg24076884", "cg24983959", "cg25775449",
                                                   "cg26284390", "cg26900154", "cg27039606", "cg27195917", "cg27195917",
                                                   "cg08878304", "cg16970232", "cg10106388", "cg11911951", "cg05467458",
                                                   "cg06584407", "cg13097816", "cg21728447", "cg05440289", "cg17115258",
                                                   "cg17452384", "cg04008901", "cg08089301", "cg13771579", "cg15878317",
                                                   "cg07899016", "cg20256783", "cg21634602", "cg01745510", "cg14074117"]]"""

    return dna_methylation_data


def get_genes_by_case_id(case_id):
    gene_table = pd.read_csv('./data/RNA_60gene.csv', encoding='gbk')
    genes = gene_table[gene_table['case_id'] == case_id]
    """pattern = r'\d+'
    id = re.findall(pattern, case_id)   
    id = list(map(lambda x: int(x), id))
    index = int(id[0])-1"""

    #genes_list = genes.loc[genes.index, ["ZNF597", "SNX4", "PHLDB2", "KLF3", "AMBRA1", "FBXO41", "COA8", "COG2","HSPA8", "CASP3"]]
    #genes_list = genes.loc[genes.index, ["TSPAN6", "TNMD", "STPG1", "SEMA3F", "SCYL3", "NIPAL3", "NFYA", "LAS1L", "GCLC", "FUCA2", "FGR", "ENPP4", "DPM1", "CFH", "C1orf112"]]

    #genes_list = genes.loc[genes.index, ["ARMC12", "CRACR2A", "FAM171B", "ITIH3", "MYOCD", "NPW", "PARP9", "PCDH10", "PCGF5", "RAPSN", "SH2D1B", "SMARCC1", "SWI5", "TNFSF11", "TTC39C"]]
    # SVM 20genes
    #genes_list = genes.loc[genes.index, ["C10orf67", "CASP7", "CKLF", "CLEC4C", "CLNK", "CLNK", "FLT3", "NDUFA4", "NEK4", "PARP9", "PCGF5", "PNRC2", "SDHAF4", "SLC1A4", "SLC23A1", "SNX4", "TRANK1", "UBE2L6", "USP4", "WDR82"]]
    # 40
    """genes_list = genes.loc[
        genes.index, ["USP4", "PNRC2", "SLC23A1", "EIF4E3", "SNX4", "PARP9", "WDR82", "SLC1A4", "NEK4", "CLNK",
                      "FLT3", "UBE2L6", "CKLF", "CASP7", "PCGF5", "TRANK1", "SDHAF4", "NDUFA4", "CLEC4C", "C10orf67",
                      "DENND6A", "SWI5", "RNF213", "DTX3L", "DBR1", "ACTR8", "TNFSF11", "SIAH2", "PLAAT4", "SHQ1",
                      "ZNFX1", "CCR5", "FKBP1C", "GPR18", "PARP14", "RYBP", "TRAFD1", "TLR10", "RTL8C", "LAX1"]]"""
    #60
    genes_list = genes.loc[
        genes.index, ["ACTR8", "ALAS2", "ATXN7", "C10orf67", "CASP7", "CCDC175", "CCR5", "CD160", "CD226", "CKLF",
                      "CLEC17A","CLEC4C", "CLNK", "COA8", "DBR1", "DENND6A", "DOCK10", "DTX3L", "EIF4E3", "FKBP1C",
                      "FLT3", "GPR18", "IL26", "KCNA3", "LAX1", "NDUFA4", "NEK4", "PARP14", "PARP9", "PLAAT4",
                      "PNRC2", "PREX1", "QRICH1", "RNF213", "RTL8C", "RYBP", "SAMD9L", "SDHAF4", "SFMBT1", "SH2D1B", 
                      "SHQ1", "SIAH2", "SLC12A3", "SLC1A4", "SLC23A1", "SLMAP", "SNX4", "SWI5", "TLR10", "TNFSF11",
                      "TRAFD1", "TRANK1", "TUBA3E", "UBE2L6", "USP4", "VKORC1", "WDR82", "ZNFX1", "PCGF5", "SRRM1"]]
    #80
    """genes_list = genes.loc[
            genes.index, ["ACTR8", "ALAS2", "ATXN7", "C10orf67", "CASP7", "CCDC175", "CCR5", "CD160", "CD226", "CKLF",
                          "CLEC17A","CLEC4C", "CLNK", "COA8", "DBR1", "DENND6A", "DOCK10", "DTX3L", "EIF4E3", "FKBP1C",
                          "FLT3", "GPR18", "IL26", "KCNA3", "LAX1", "NDUFA4", "NEK4", "PARP14", "PARP9", "PLAAT4",
                          "PNRC2", "PREX1", "QRICH1", "RNF213", "RTL8C", "RYBP", "SAMD9L", "SDHAF4", "SFMBT1", "SH2D1B",
                          "SHQ1", "SIAH2", "SLC12A3", "SLC1A4", "SLC23A1", "SLMAP", "SNX4", "SWI5", "TLR10", "TNFSF11",
                          "TRAFD1", "TRANK1", "TUBA3E", "UBE2L6", "USP4", "VKORC1", "WDR82", "ZNFX1", "PCGF5", "SRRM1",
                          "CLEC9A", "EBLN2", "DDX60", "NFATC2", "NDRG1", "PARD6B", "ZNF287", "GRPR", "ENPP7", "FAM216B",
                          "RGS20", "RAB8B", "BIRC3", "CDC42SE2", "IFNLR1", "SHOC2", "WFDC1", "PDE12", "HPS5", "BCL2L11"]]"""

    return genes_list


def get_clinicals_by_case_id(case_id):
    clinicals_table = pd.read_csv('./data/Clinical_207.csv')
    clinicals = clinicals_table[clinicals_table['case_id'] == case_id]
    """pattern = r'\d+'
    id = re.findall(pattern, slide_id)
    id = list(map(lambda x: int(x), id))
    index = int(id[0]) - 1"""
    clinicals_list = clinicals.loc[clinicals.index, ["T_stage", "N_stage", "M_stage", "Stage", "ER", "PR"]]
    #clinicals_list = clinicals.loc[clinicals.index, ["T_stage", "N_stage", "M_stage", "Stage"]]
    clinicals_age = clinicals.loc[clinicals.index, "age"]
    return clinicals_list, clinicals_age


def max_min(age):
    clinicals_table = pd.read_csv('./data/Clinical_207.csv')
    clinicals_age = clinicals_table['age']

    max = 0
    min = 100
    for i in range(len(clinicals_age)):
        clinicals = clinicals_table[clinicals_table['slide_id'] == 'slide_{}'.format(i+1)]
        c_age = clinicals.loc[i, 'age']
        if (c_age>max):
            max = c_age
        if (c_age<min):
            min = c_age

    finial_age = (age-min) / (max - min)
    return finial_age




