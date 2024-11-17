import pandas as pd


def get_bank():

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
    
    colonnes_francais = [
        "Statut_compte_courant", "Duree_mois", "Historique_credit", "Objectif", "Montant_credit",
        "Compte_epargne", "Emploi_actuel_depuis", "Taux_versement", "Statut_personnel_sexe",
        "Autres_debiteurs", "Residence_depuis", "Propriete", "Age_annees", "Autres_plans_versement",
        "Logement", "Nb_credits_existant", "Travail", "Nb_personnes_a_charge", "Telephone", "Travailleur_etranger", "Cible"
    ]
    
    data = pd.read_csv(url, header=None, names=colonnes_francais)
    
    data.to_csv('ban_data.csv',index=False)
    
    
def get_bank():

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
    
    colonnes_francais = [
        "Statut_compte_courant", "Duree_mois", "Historique_credit", "Objectif", "Montant_credit",
        "Compte_epargne", "Emploi_actuel_depuis", "Taux_versement", "Statut_personnel_sexe",
        "Autres_debiteurs", "Residence_depuis", "Propriete", "Age_annees", "Autres_plans_versement",
        "Logement", "Nb_credits_existant", "Travail", "Nb_personnes_a_charge", "Telephone", "Travailleur_etranger", "Cible"
    ]
    
    data = pd.read_csv(url, header=None, names=colonnes_francais)
    
    data.to_csv('ban_data.csv',index=False)
