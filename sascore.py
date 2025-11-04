from __future__ import print_function, division
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdMolDescriptors

import math
import pickle
import os, gzip
# rdBase.DisableLog('rdApp.error')
import log

class SAscore():
    """
    Calculation of synthetic accessibility score as described in:
    Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
    Peter Ertl and Ansgar Schuffenhauer
    Journal of Cheminformatics 1:8 (2009)
    http://www.jcheminf.com/content/1/1/8
    """
    def __init__(self):
        global _fscores
        _fscores = None

    def __call__(self, smile):
        if _fscores is None:
            self.readFragmentScores()

        m = Chem.MolFromSmiles(smile)
        if not m:
            return 0.0

        try:
            # fragment score
            log.debug(f"Calculating SAscore for molecule: {smile}")
            fp  = rdMolDescriptors.GetMorganFingerprint(m, radius=2)  # sparse count FP
            fps = fp.GetNonzeroElements()
            nf  = sum(fps.values())
            score1 = (sum(_fscores.get(bitId, -4) * v for bitId, v in fps.items()) / nf) if nf > 0 else 0.0

            # features score
            nAtoms = m.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
            ri = m.GetRingInfo()
            nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
            nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
            nMacrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = math.log10(2) if nMacrocycles > 0 else 0.0  # paper differs

            score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

            # symmetry correction (not in original paper)
            score3 = 0.0
            if len(fps) > 0 and nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * 0.5

            sascore = score1 + score2 + score3

            # map to 1..10, then convert to “lower is better”
            min_score, max_score = -4.0, 2.5
            sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
            if sascore > 8.:
                sascore = 8. + math.log(sascore + 1. - 9.)
            if sascore > 10.:
                sascore = 10.0
            elif sascore < 1.:
                sascore = 1.0

            sascore = math.exp(1 - sascore)  # minimize SA
            return float(sascore)

        except Exception as e:
            log.error(f"SAscore failed for {smile}: {e}")
            return 0.0

    def readFragmentScores(self, name='fpscores'):
        """Load pickled fragment scores (gzip). Looks next to this file."""
        global _fscores
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, f"{name}.pkl.gz")
        with gzip.open(path, 'rb') as f:
            raw = pickle.load(f)
        out = {}
        for row in raw:
            for j in range(1, len(row)):
                out[row[j]] = float(row[0])
        _fscores = out
