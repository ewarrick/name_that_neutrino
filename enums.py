from collections import defaultdict
from enum import Enum, auto

from icecube.dataclasses import I3Particle
from icecube import tableio

class classification(tableio.enum3.enum):
    """Convenience class for labelling the classifications"""
    unclassified = 0
    throughgoing_track = 1
    starting_track = 2
    stopping_track = 3
    skimming_track = 4
    contained_track = 5
    contained_em_hadr_cascade = 6
    contained_hadron_cascade = 7
    uncontained_cascade = 8
    glashow_starting_track = 9
    glashow_electron = 10
    glashow_tau_double_bang = 11
    glashow_tau_lollipop = 12
    glashow_hadronic = 13
    throughgoing_tau = 14
    skimming_tau = 15
    double_bang = 16
    lollipop = 17
    inverted_lollipop = 18
    throughgoing_bundle = 19
    stopping_bundle = 20
    tau_to_mu = 21

    # no skimming bundle for now!
    # skimming bundle = skimming track


class interaction_types(Enum):
    """Convenience class for labelling neutrino interactions"""
    nue_nc = auto()
    nue_cc = auto()
    numu_nc = auto()
    numu_cc = auto()
    nutau_nc = auto()
    nutau_cc = auto()
    gr_leptonic_e = auto()
    gr_leptonic_mu = auto()
    gr_leptonic_tau = auto()
    gr_hadronic = auto()
    corsika = auto()

class containments_types(Enum):
    """Convenience class for labelling event containment types"""
    no_intersect = auto()
    throughgoing = auto()  # For tracks only
    contained = auto()  # Cascade or track that starts and stops in detector volume
    tau_to_mu = auto() # Special case for a contained tau that decays into a muon
    starting = auto()
    stopping = auto()
    decayed = auto()
    throughgoing_bundle = auto()
    stopping_bundle = auto()


    # no skimming bundle for now!
    # skimming bundle = skimming track


# Interactions that result in cascade events
cascade_interactions = [
    interaction_types.nue_nc,
    interaction_types.numu_nc,
    interaction_types.nutau_nc,
    interaction_types.nue_cc,
    interaction_types.gr_leptonic_e,
    interaction_types.gr_hadronic,
]

# Interactions that result in (muon) tracks
track_interactions = [interaction_types.numu_cc, interaction_types.gr_leptonic_mu]

# Interactions that produce taus
tau_interactions = [interaction_types.nutau_cc, interaction_types.gr_leptonic_tau]

# NC interactions
nc_interactions = [
    interaction_types.nue_nc,
    interaction_types.numu_nc,
    interaction_types.nutau_nc]

# Mapping to convert the NuGen interaction type
nugen_int_t_mapping = {
    (int(I3Particle.NuE), 1): interaction_types.nue_cc,
    (int(I3Particle.NuEBar), 1): interaction_types.nue_cc,
    (int(I3Particle.NuMu), 1): interaction_types.numu_cc,
    (int(I3Particle.NuMuBar), 1): interaction_types.numu_cc,
    (int(I3Particle.NuTau), 1): interaction_types.nutau_cc,
    (int(I3Particle.NuTauBar), 1): interaction_types.nutau_cc,

    (int(I3Particle.NuE), 2): interaction_types.nue_nc,
    (int(I3Particle.NuEBar), 2): interaction_types.nue_nc,
    (int(I3Particle.NuMu), 2): interaction_types.numu_nc,
    (int(I3Particle.NuMuBar), 2): interaction_types.numu_nc,
    (int(I3Particle.NuTau), 2): interaction_types.nutau_nc,
    (int(I3Particle.NuTauBar), 2): interaction_types.nutau_nc,
}


# Mapping from interaction type and containment type to classification
class_mapping = {
    (interaction_types.gr_leptonic_e, containments_types.contained): classification.glashow_electron,
    (interaction_types.gr_leptonic_e, containments_types.no_intersect): classification.uncontained_cascade,

    (interaction_types.gr_leptonic_mu, containments_types.no_intersect): classification.skimming_track,
    (interaction_types.gr_leptonic_mu, containments_types.throughgoing): classification.throughgoing_track,
    (interaction_types.gr_leptonic_mu, containments_types.contained): classification.glashow_starting_track,
    (interaction_types.gr_leptonic_mu, containments_types.starting): classification.glashow_starting_track,
    (interaction_types.gr_leptonic_mu, containments_types.stopping): classification.stopping_track,

    (interaction_types.gr_leptonic_tau, containments_types.no_intersect): classification.skimming_tau,
    (interaction_types.gr_leptonic_tau, containments_types.throughgoing): classification.throughgoing_tau,
    (interaction_types.gr_leptonic_tau, containments_types.contained): classification.glashow_tau_double_bang,
    (interaction_types.gr_leptonic_tau, containments_types.starting): classification.glashow_tau_lollipop,
    (interaction_types.gr_leptonic_tau, containments_types.stopping): classification.inverted_lollipop,

    (interaction_types.gr_hadronic, containments_types.contained): classification.glashow_hadronic,
    (interaction_types.gr_hadronic, containments_types.no_intersect): classification.uncontained_cascade,

    (interaction_types.numu_cc, containments_types.no_intersect): classification.skimming_track,
    (interaction_types.numu_cc, containments_types.throughgoing): classification.throughgoing_track,
    (interaction_types.numu_cc, containments_types.contained): classification.contained_track,
    (interaction_types.numu_cc, containments_types.starting): classification.starting_track,
    (interaction_types.numu_cc, containments_types.stopping): classification.stopping_track,

    (interaction_types.nutau_cc, containments_types.no_intersect): classification.skimming_tau,
    (interaction_types.nutau_cc, containments_types.throughgoing): classification.throughgoing_tau,
    (interaction_types.nutau_cc, containments_types.contained): classification.double_bang,
    (interaction_types.nutau_cc, containments_types.starting): classification.lollipop,
    (interaction_types.nutau_cc, containments_types.tau_to_mu): classification.tau_to_mu,

    # If the tau is stopping, we see an incoming tau, the cascade from the tau decay and an outgoing muon.
    (interaction_types.nutau_cc, containments_types.stopping): classification.inverted_lollipop,

    (interaction_types.nue_cc, containments_types.contained): classification.contained_em_hadr_cascade,
    (interaction_types.nue_cc, containments_types.no_intersect): classification.uncontained_cascade,

    # Corsika single muons
    (interaction_types.corsika, containments_types.throughgoing): classification.throughgoing_track,
    (interaction_types.corsika, containments_types.no_intersect): classification.skimming_track,
    (interaction_types.corsika, containments_types.stopping): classification.stopping_track,

    # Corsika bundles (more than two muons reaching the detector)
    (interaction_types.corsika, containments_types.throughgoing_bundle): classification.throughgoing_bundle,
    (interaction_types.corsika, containments_types.stopping_bundle): classification.stopping_bundle,
    (interaction_types.corsika, containments_types.no_intersect): classification.skimming_track,



}

# add nc mappings
for int_t in nc_interactions:
    class_mapping[(int_t, containments_types.no_intersect)] = classification.uncontained_cascade
    class_mapping[(int_t, containments_types.contained)] = classification.contained_hadron_cascade
