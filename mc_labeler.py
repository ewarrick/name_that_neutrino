#mc_labeler.py script
from enums import (
    cascade_interactions,
    classification,
    class_mapping,
    containments_types,
    interaction_types,
    nugen_int_t_mapping,
    track_interactions,
    tau_interactions,
)

import numpy as np
import warnings
from icecube import dataclasses, dataio, MuonGun, icetray
from icecube.dataclasses import I3Particle
from I3Tray import I3Units

try:
    from icecube import LeptonInjector

    LI_FOUND = True
except:
    LI_FOUND = False


# Convenience collections
negative_charged_leptons = [I3Particle.EMinus, I3Particle.MuMinus, I3Particle.TauMinus]
positive_charged_leptons = [I3Particle.EPlus, I3Particle.MuPlus, I3Particle.TauPlus]
all_charged_leptons = negative_charged_leptons + positive_charged_leptons

muon_types = [I3Particle.MuMinus, I3Particle.MuPlus]
tau_types = [I3Particle.TauMinus, I3Particle.TauPlus]
electron_types = [I3Particle.EMinus, I3Particle.EPlus]

neutrino_types = [I3Particle.NuE, I3Particle.NuMu, I3Particle.NuTau]
anti_neutrino_types = [I3Particle.NuEBar, I3Particle.NuMuBar, I3Particle.NuTauBar]
all_neutrinos = neutrino_types + anti_neutrino_types

electron_neutrinos = [I3Particle.NuE, I3Particle.NuEBar]
muon_neutrinos = [I3Particle.NuMu, I3Particle.NuMuBar]
tau_neutrinos = [I3Particle.NuTau, I3Particle.NuTauBar]

# from clsim
cascade_types = [
    I3Particle.Neutron,
    I3Particle.Hadrons,
    I3Particle.Pi0,
    I3Particle.PiPlus,
    I3Particle.PiMinus,
    I3Particle.K0_Long,
    I3Particle.KPlus,
    I3Particle.KMinus,
    I3Particle.PPlus,
    I3Particle.PMinus,
    I3Particle.K0_Short,
    I3Particle.EMinus,
    I3Particle.EPlus,
    I3Particle.Gamma,
    I3Particle.Brems,
    I3Particle.DeltaE,
    I3Particle.PairProd,
    I3Particle.NuclInt,
]


class MCLabeler(icetray.I3Module):
    def __init__(self, context):
        super().__init__(context)
        gcd = '/home/icecube/Desktop/eliz_zooniverse/icecubezooniverseproj_ver3/i3_files/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'
        open_gcd = dataio.I3File(gcd)
        open_gcd.rewind()
        frame1 = open_gcd.pop_frame(icetray.I3Frame.Geometry)
        i3geo = frame1['I3Geometry']

        self.AddParameter("gcd", "Path of GCD File. If none use g frame", i3geo
)
        self.AddParameter(
            "cr_muon_padding",
            "Padding for CR muons. Increase to count muons passing further out. ",
            150 * I3Units.m,
        )
        self.AddParameter(
            "det_hull_padding",
            "Padding for the detector hull for calculating containment.",
            0 * I3Units.m,
        )
        self.AddParameter(
            "mcpe_pid_map_name",
            "Name of the I3MCPESeriesMapParticleIDMap. Set to `None` to disable background MCPE counting."
            "Note: Naive MCPE downsampling will render I3MCPESeriesMapParticleIDMap useles",
            "I3MCPESeriesMapParticleIDMap",
        )
        self.AddParameter(
            "mcpe_map_name",
            "Name of the I3MCPESeriesMap",
            "I3MCPulseSeriesMap",
        )
        self.AddParameter("mctree_name", "Name of the I3MCTree", "SignalI3MCTree")
        self.AddParameter(
            "bg_mctree_name",
            "Name of the background I3MCTree. (Change if coincident events are in a"
            " separate MCTree)",
            "I3MCTree",
        )
        self.AddParameter(
            "event_properties_name", "Name of the LI EventProperties.", None
        )
        self.AddParameter("weight_dict_name", "Name of the I3MCWeightDict", 'I3MCWeightDict')
        self.AddParameter(
            "corsika_weight_map_name", "Name of the CorsikaWeightMap", None
        )
        self.AddParameter("key_postfix", "Postfix for the keys stored in the frame", "")

        self._surface = None
        self._surface_cr = None

    def Configure(self):
        self._geo = self.GetParameter("gcd")
        self._cr_muon_padding = self.GetParameter("cr_muon_padding")
        self._det_hull_padding = self.GetParameter("det_hull_padding")
        self._mcpe_pid_map_name = self.GetParameter("mcpe_pid_map_name")
        self._mcpe_map_name = self.GetParameter("mcpe_map_name")
        self._mctree_name = self.GetParameter("mctree_name")
        self._bg_mctree_name = self.GetParameter("bg_mctree_name")
        self._event_properties_name = self.GetParameter("event_properties_name")
        self._weight_dict_name = self.GetParameter("weight_dict_name")
        self._corsika_weight_map_name = self.GetParameter("corsika_weight_map_name")
        self._key_postfix = self.GetParameter("key_postfix")

        if (self._event_properties_name is not None) + (
            self._weight_dict_name is not None) + (self._corsika_weight_map_name is not None) != 1:
            raise RuntimeError(
                "Set only one of event_properties_name, weight_dict_name and corsika_weight_map_name"
            )

        self._is_li = self._event_properties_name is not None
        if self._is_li and not LI_FOUND:
            raise RuntimeError(
                "Simulation is LeptonInjector but couldn't import LeptonInjector."
            )
        self._is_corsika = self._corsika_weight_map_name is not None

    def Geometry(self, frame):
        if self._geo is None:
            self._geo = frame["I3Geometry"]
        self.PushFrame(frame)

    @staticmethod
    def find_neutrinos(tree):
        return tree.get_filter(
            lambda p: (p.type in all_neutrinos) and np.isfinite(p.length)
        )

    @staticmethod
    def get_inice_neutrino(tree, is_li):
        #print(tree.get_primaries())
        neutrino_primary = [p for p in tree.get_primaries() if p.type in all_neutrinos] #here is fix
        #print(len(neutrino_primary))
        if len(neutrino_primary) != 1:
            return 1
            #raise RuntimeError("Found more or less than one primary neutrino")
        neutrino_primary = neutrino_primary[0]

        if is_li:
            # Assume that LI only inserts the final, in-ice neutrino
            return neutrino_primary


        # Not LI. Find highest energy in-ice neutrino
        in_ice_nu = tree.get_best_filter(
            lambda p: (p.location_type == I3Particle.InIce) and
                      (p.type in all_neutrinos) and
                      tree.is_in_subtree(neutrino_primary, p),
            lambda p1, p2: p1.energy > p2.energy)


        # if not in_ice_nu:
        # For some reason, NuGen sometimes marks non-final neutrinos as in-ice.
        # As a work-around find the highest energy in-ice particle in the
        # subtree of the neutrino primary and work back from there
        in_ice = tree.get_best_filter(
            lambda p: (p.location_type == I3Particle.InIce)
            and ((p.type in all_charged_leptons) or (p.type == I3Particle.Hadrons))
            and tree.is_in_subtree(neutrino_primary, p),
            lambda p1, p2: p1.energy > p2.energy,
        )

        def parent_nu(part, tree):
            """Recursively find the parent neutrino"""
            if part.type in all_neutrinos:
                return part
            parent = tree.parent(part)
            return parent_nu(parent, tree)

        in_ice_nu = parent_nu(in_ice, tree)

        # Sanity check

        nu_children = tree.children(in_ice_nu)

        subnu = [
            p
            for p in nu_children
            if (p.type in all_neutrinos) and p.location_type == I3Particle.InIce
        ]
        if subnu and tree.children(subnu[0]):
            print("Warning: found two in-ice neutrinos, trying child particle")
            return subnu[0]

        return in_ice_nu

    @staticmethod
    def get_corsika_muons(tree):
        primaries = [p for p in tree.get_primaries() if p.type not in all_neutrinos]
        muons = [
            p
            for primary in primaries
            for p in tree.children(primary)
            if p.type in muon_types
        ]

        return muons

    @staticmethod
    def get_containment(
        p, surface, decayed_before_type=containments_types.no_intersect
    ):
        """
        Determine containment type for particle `p`.
        if `p` is a track, the `decayed_before_type` allows specifiying the
        containment type of particles that would intersect with the
        surface, but decay before entering.
        """

        intersections = surface.intersection(p.pos, p.dir)

        if not np.isfinite(intersections.first):
            return containments_types.no_intersect

        if p.is_cascade:
            if intersections.first <= 0 and intersections.second > 0:
                return containments_types.contained
            return containments_types.no_intersect

        if p.is_track:
            # Check if starting or contained
            if intersections.first <= 0 and intersections.second > 0:
                if p.length <= intersections.second:
                    return containments_types.contained
                return containments_types.starting

            # Check if throughgoing or stopping
            if intersections.first > 0 and intersections.second > 0:
                if p.length <= intersections.first:
                    return decayed_before_type
                if p.length > intersections.second:
                    return containments_types.throughgoing
                else:
                    return containments_types.stopping
        return containments_types.no_intersect

    @staticmethod
    def get_neutrino_interaction_type_li(prop):
        # Test for leptonic Glashow
        if (
            (prop.initialType == I3Particle.NuEBar)
            and (prop.finalType1 in negative_charged_leptons)
            and (prop.finalType2 in anti_neutrino_types)
        ):
            if prop.finalType1 == I3Particle.EMinus:
                return interaction_types.gr_leptonic_e
            if prop.finalType1 == I3Particle.MuMinus:
                return interaction_types.gr_leptonic_mu
            if prop.finalType1 == I3Particle.TauMinus:
                return interaction_types.gr_leptonic_tau

        # Test for hadronic Glashow
        if (
            (prop.initialType == I3Particle.NuEBar)
            and (prop.finalType1 == I3Particle.Hadrons)
            and (prop.finalType2 == I3Particle.Hadrons)
        ):
            return interaction_types.gr_hadronic

        # Test for CC
        if prop.finalType1 in all_charged_leptons:
            if prop.initialType in electron_neutrinos:
                return interaction_types.nue_cc
            if prop.initialType in muon_neutrinos:
                return interaction_types.numu_cc
            if prop.initialType in tau_neutrinos:
                return interaction_types.nutau_cc

        # Test for NC
        if prop.finalType1 in all_neutrinos:
            if prop.initialType in electron_neutrinos:
                return interaction_types.nue_nc
            if prop.initialType in muon_neutrinos:
                return interaction_types.numu_nc
            if prop.initialType in tau_neutrinos:
                return interaction_types.nutau_nc

        raise RuntimeError(
            "Unknown interaction type: {} -> {} + {}".format(
                prop.initialType, prop.finalType1, prop.finalType2
            )
        )

    @staticmethod
    def get_neutrino_interaction_type_nugen(wdict, tree):
        int_t = wdict["InteractionType"]
        nutype = wdict["InIceNeutrinoType"]
        neutrino = MCLabeler.get_inice_neutrino(tree, is_li=False)

        if neutrino is None:
            return None

        children = tree.children(neutrino)
        if len(children) != 2:

            raise RuntimeError(
                "Neutrino interaction with more or less than two children."
            )

        if int_t != 3:
            return nugen_int_t_mapping[(nutype, int_t)]

        if int_t == 3:
            # GR.
            if (children[0].type == I3Particle.Hadrons) and (
                children[1].type == I3Particle.Hadrons
            ):
                return interaction_types.gr_hadronic
            if (children[0].type in electron_types) or (
                children[1].type in electron_types
            ):
                return interaction_types.gr_leptonic_e
            if (children[0].type in muon_types) or (children[1].type in muon_types):
                return interaction_types.gr_leptonic_mu
            if (children[0].type in tau_types) or (children[1].type in tau_types):
                return interaction_types.gr_leptonic_tau
        raise RuntimeError(
            "Unknown interaction type: {} -> {} + {} (Nugen type {})".format(
                neutrino.type, children[0].type, children[1].type, int_t
            )
        )

    def _classify_neutrinos(self, frame):

        tree = frame[self._mctree_name]
        if self._is_li:
            prop = frame[self._event_properties_name]
            int_t = self.get_neutrino_interaction_type_li(prop)
        else:
            wdict = frame[self._weight_dict_name]
            int_t = self.get_neutrino_interaction_type_nugen(wdict, tree)

        in_ice_neutrino = self.get_inice_neutrino(tree, self._is_li)

        if in_ice_neutrino is not None:

            children = tree.children(in_ice_neutrino)
            # Classify everything related to muons
            if int_t in track_interactions:
                # figure out if vertex is contained
                muons = [p for p in children if p.type in muon_types]
                if len(muons) != 1:
                    raise RuntimeError(
                        "Muon interaction with not exactly one muon child"
                    )

                containment = self.get_containment(muons[0], self._surface)

            # Classify everything related to cascades

            elif int_t in cascade_interactions:
                cascades = [p for p in children if p.is_cascade]
                if not cascades:
                    raise RuntimeError(
                        "Found cascade-type interaction but no cascade children"
                    )
                # We can have more than one cascade, just check the first
                # TODO: Check whether there are any pitfalls with this approach

                containment = self.get_containment(cascades[0], self._surface)

            elif int_t in tau_interactions:
                taus = [p for p in children if p.type in tau_types]
                if len(taus) != 1:
                    raise RuntimeError("Tau interaction with not exactly one tau child")

                containment = self.get_containment(taus[0], self._surface)

                # if the tau is contained, check the tau decay
                if containment == containments_types.contained:
                    tau_children = tree.children(taus[0])
                    muons = [p for p in tau_children if p.type in muon_types]
                    if len(muons) > 0:
                        # the tau decays into a muon
                        containment = containments_types.tau_to_mu

                if containment == containments_types.no_intersect:
                    # Check the containment of the resulting muon
                    tau_muons = [
                        p for p in tree.children(taus[0]) if p.type in muon_types
                    ]
                    if len(tau_muons) > 1:
                        raise RuntimeError("Tau decay with more than one muon")
                    elif len(tau_muons) == 1:
                        # We have a muon

                        muon_containment = self.get_containment(
                            tau_muons[0], self._surface
                        )
                        containment = muon_containment

                        # Since the tau is uncontained, we label the event by the topology
                        # of the muon created in the tau decay
                        int_t = interaction_types.numu_cc

            else:
                raise RuntimeError("Unknown interaction type: {}".format(int_t))
        else:
            int_t = None
            containment = None
        return int_t, containment

    def _classify_corsika(self, frame):
        """
        Classify corsika events.
        The code to distinguish bundles / single muons is not yet perfect. There might
        be edge cases, where a single muon accompanied by low-energy muons that stop far
        away from the detector is classified as skimming
        """

        tree = frame[self._mctree_name]
        corsika_muons = self.get_corsika_muons(tree)

        containments = [
            self.get_containment(
                muon, self._surface, decayed_before_type=containments_types.decayed
            )
            for muon in corsika_muons
        ]

        int_t = interaction_types.corsika

        # Check if we are dealing with a single muon.
        # Number of muons that would have intersected by decay before entering the detector
        num_decayed = len(
            [cont for cont in containments if cont == containments_types.decayed]
        )

        if num_decayed == len(containments) - 1:
            # all decayed except one. containment type is given by surviving muon
            not_decayed = [
                cont for cont in containments if cont != containments_types.decayed
            ][0]
            return int_t, not_decayed

        # at least one muon is uncontained
        if any([cont == containments_types.no_intersect for cont in containments]):
            return int_t, containments_types.no_intersect

        # All muons are stopping
        if all([cont == containments_types.stopping for cont in containments]):
            return int_t, containments_types.stopping_bundle

        # Bundle is throughgoing
        return int_t, containments_types.throughgoing_bundle

    def classify(self, frame):
        if self._mctree_name not in frame:
            raise RuntimeError("I3MCTree no found")

        if self._surface is None:
            self._surface = MuonGun.ExtrudedPolygon.from_I3Geometry(
                self._geo, self._det_hull_padding
            )
            self._surface_cr = MuonGun.ExtrudedPolygon.from_I3Geometry(
                self._geo, self._cr_muon_padding
            )

        if self._is_corsika:
            int_t, containment = self._classify_corsika(frame)
        else:
            int_t, containment = self._classify_neutrinos(frame)

        # Polyplopia
        tree = frame[self._mctree_name]
        bg_tree = frame[self._bg_mctree_name]
        poly_muons = self.get_corsika_muons(bg_tree)
        containments = [
            self.get_containment(muon, self._surface_cr) for muon in poly_muons
        ]

        n_stop_through = sum(
            [
                1
                for cont in containments
                if cont
                in [containments_types.stopping, containments_types.throughgoing]
            ]
        )

        mcpe_from_muons = 0
        mcpe_from_muons_charge = 0

        # Sadly some simulations break the MCPEID map, so give useres the chance to skip
        if self._mcpe_pid_map_name is not None and self._mcpe_pid_map_name in frame:
            # Also collect MCPE from CR muons
            poly_muon_ids = [p.id for p in poly_muons]
            # Most MCPE will be caused by daughter particles of the muon
            poly_muon_ids += [ch.id for p in poly_muons for ch in tree.children(p)]

            mcpe_series_map = frame[self._mcpe_map_name]
            if self._mcpe_map_name in frame:
                # Collect the total mcpe charge from CR muons
                for omkey, idmap in frame[self._mcpe_pid_map_name]:

                    if omkey not in mcpe_series_map:
                        warnings.warn("Couldn't find OMKey in MCPESeriesMap")
                    else:
                        mcpe_series = mcpe_series_map[omkey]
                        for pmid in poly_muon_ids:
                            # loop through the PIDs
                            if pmid in idmap.keys():
                                mcpe_indices = idmap[pmid]
                                mcpe_from_muons_charge += sum(
                                    [mcpe_series[i].npe for i in mcpe_indices]
                                )

                                mcpe_from_muons += len(mcpe_indices)

        return (
            class_mapping.get((int_t, containment), classification.unclassified),
            n_stop_through,
            mcpe_from_muons,
            mcpe_from_muons_charge,
        )

    def DAQ(self, frame):
        if self._geo is None:
            raise RuntimeError("No geometry information found")
        classif, n_coinc, bg_mcpe, bg_mcpe_charge = self.classify(frame)
        frame["classification" + self._key_postfix] = icetray.I3Int(int(classif))
        frame["classification_label" + self._key_postfix] = dataclasses.I3String(
            classif.name
        )
        frame["coincident_muons" + self._key_postfix] = icetray.I3Int(n_coinc)
        frame["bg_muon_mcpe" + self._key_postfix] = icetray.I3Int(bg_mcpe)
        frame["bg_muon_mcpe_charge" + self._key_postfix] = dataclasses.I3Double(
            bg_mcpe_charge
        )
        self.PushFrame(frame)
