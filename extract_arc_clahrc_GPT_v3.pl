
#!/usr/bin/env perl
# Tightened ARC/CLAHRC extraction with DEBUG CSV:
# - Explicit-first matching per chunk
# - Conservative fallback location matching only when no explicit ARC in chunk
# - Guardrails for common false positives (Kent surname, EM/LC initials, Imperial, Essex-in-Wessex, Durham, etc.)
# - Writes arc_match_debug.csv with rich diagnostics to tune patterns

# PREVIOUS VERSION
# extract_arc_clahrc_GPT_revised.pl (rewritten: anchor-window + CRN/site-list suppression + debug logging)
# - Uses anchor-window matching to prevent long paragraphs from generating many ARC matches
# - Suppresses matches inside CRN lists / Clinical Research Network text
# - Suppresses matches inside numbered recruitment site lists / tables
# - Adds arc_match_debug.csv to audit why each match fired
#
# NOTE: Uses UTF-8 IO, but avoids smart punctuation in code/comments.
# IL Updated 27/Jan/2026 13:04

# PREVIOUS VERSION:
# script written by ChatGPT v5.2 to extract ARC and CLAHRC info from WoS address and funding/acknowledgement texts
# UPDATED VERSION (27/Jan/2026):
# - Prevents cross-field "licensing" (a general hit in FU no longer enables direct location matches in ADDR)
# - Disables addr:direct_loc to avoid multicentre address lists generating many false ARC region hits
# - Funding/ack matching is chunked on ';' and requires an infra anchor + location cue in the SAME chunk
# - Uses a stricter infra anchor for proximity/direct matching (not bare \bARC\b)
# - Fixes paper_flags.csv header to include specific_reasons (prevents column shift)
# IL this is a rewritten version based on the previous "extract_arc_clahrc.pl" script which was generating multiple false positives
# IL 27/Jan/2026 12:17

# 3/Feb/2026 17:30 - rewritten by ChatGPT
#!/usr/bin/env perl
# Tightened ARC/CLAHRC extraction with DEBUG CSV (rewritten)
# Key changes in this version:
# - "West" is treated safely (will NOT match North West / South West / West Midlands / NW London / NW Coast / SW Peninsula)
# - ARC East of England fallback no longer fires on Cambridge/Cambridgeshire/Peterborough etc (too leaky)
# - ARC North East and North Cumbria fallback no longer fires on generic "North East"
# - ARC North Thames explicit patterns strengthened
# - Post-match resolver drops spillover fallback hits when stronger evidence exists
# - Keeps debug CSV rich and adds "blocked_*" rules for auditing

use strict;
use warnings;
use open qw(:std :encoding(UTF-8));
binmode(STDOUT, ":encoding(UTF-8)");
binmode(STDERR, ":encoding(UTF-8)");

use utf8;
use Text::CSV;

# ----------------------------
# CONFIG
# ----------------------------
my $INPUT_CSV  = shift @ARGV || "input.csv";
my $OUT_FLAGS  = "paper_flags.csv";
my $OUT_LONG   = "paper_infra_long.csv";
my $OUT_DEBUG  = "arc_match_debug.csv";

my $COL_DOI   = "DOI";
my $COL_PMID  = "PMID";
my $COL_ADDR  = "AuthorAddresses";
my $COL_FU    = "FundingAcknowledgements";

# ----------------------------
# Helpers
# ----------------------------
sub norm_text {
  my ($s) = @_;
  $s = "" unless defined $s;
  $s =~ s/\r\n?/\n/g;
  # Normalize common unicode punctuation to ASCII-ish
  $s =~ s/[\x{2010}\x{2011}\x{2012}\x{2013}\x{2014}\x{2212}]/-/g;
  $s =~ s/[\x{2018}\x{2019}]/'/g;
  $s =~ s/[\x{201C}\x{201D}]/"/g;
  $s =~ s/\s+/ /g;
  $s =~ s/^\s+|\s+$//g;
  return $s;
}

sub get_paper_id {
  my ($row) = @_;
  my $doi  = $row->{$COL_DOI}  // "";
  my $pmid = $row->{$COL_PMID} // "";
  $doi  =~ s/^\s+|\s+$//g;
  $pmid =~ s/^\s+|\s+$//g;
  return $doi ne "" ? $doi : ($pmid ne "" ? "PMID:$pmid" : "NOID");
}

sub split_chunks {
  my ($s) = @_;
  $s = "" unless defined $s;
  my @chunks = split(/\s*;\s*/, $s);
  return @chunks;
}

sub has_kent_surname {
  my ($s) = @_;
  return ($s =~ /\bKent,\s*[A-Z]/) ? 1 : 0;
}

sub has_birmingham_brc_only {
  my ($s) = @_;
  return ($s =~ /\bBirmingham\s+Biomedical\s+Research\s+Centre\b/i) ? 1 : 0;
}

sub shorten {
  my ($s, $n) = @_;
  $n ||= 240;
  $s = "" unless defined $s;
  $s =~ s/\s+/ /g;
  return length($s) <= $n ? $s : substr($s, 0, $n) . "...";
}

# ----------------------------
# 1) GENERAL infra signal (broad)
# ----------------------------
my $RE_GENERAL = qr/
  \bARC\b
  | \bCLAHRC\b
  | \bCLA[- ]?HRC\b
  | \bPenARC\b
  | \bPenCLAHRC\b
  | Applied\s+Research\s+Collaboration
  | Appl\s+Res\s+Collaborat
  | Collaboration\s+for\s+Applied\s+Health\s+Research\s+and\s+Care
  | (?:Collaborations?\s+for|Collaborative)\s+Leadership
      (?:\s+(?:in|&|and))?
      \s+(?:Applied\s+)?Health\s+Research
      (?:\s+(?:&|and)\s+Care)?
/xmi;

# Strong infra anchor (must be present in chunk to do anything)
my $RE_INFRA_ANCHOR = qr/
  \bNIHR\b.{0,15}\bARC\b
  | \bNIHR\b.{0,20}\bCLAHRC\b
  | NIHR\s+Applied\s+Research\s+Collaboration
  | Applied\s+Research\s+Collaboration
  | Collaboration\s+for\s+Applied\s+Health\s+Research\s+and\s+Care
  | \bCLAHRC\b
  | \bCLA[- ]?HRC\b
  | \bPenARC\b
  | \bPenCLAHRC\b
/xmi;

# ----------------------------
# 2) EXCLUSIONS (lightweight)
# ----------------------------
my $RE_EXCL_ADDR = qr/
  \bNHMRC-ARC\b
  | \bARC\s+Sante\b
  | \bAustralian\s+Research\s+Council\b
  | \bAdvanced\s+Research\s+Computing\b
  | \bAgricultural\s+Research\s+C(?:ouncil)?\b
  | \bARCS\b(?!outh\s*London\b)
/xmi;

my $RE_EXCL_FU = qr/
  \bAustralian\s+Research\s+Council\b
  | \bAgricultural\s+Research\s+C(?:ouncil)?\b
  | \bAdvanced\s+Research\s+Computing\b
/xmi;

# ----------------------------
# 3) Safe "West" token (prevents West being inferred from North West / South West / West Midlands etc.)
# ----------------------------
my $RE_WEST_SOLO = qr/
  (?<!North[\s-])
  (?<!South[\s-])
  \bWest\b
  (?!\s+(?:Midlands|Peninsula|Coast|London)\b)
/xmi;

# ----------------------------
# 4) ARC explicit name patterns (HIGH PRECISION)
# ----------------------------
my %ARC_EXPLICIT = (

  # ARC West (safe West token; excludes North West / South West / West Midlands / etc.)
  'ARC West' => qr/
    \bNIHR\s+(?:ARC|CLAHRC)\s*$RE_WEST_SOLO
    | \b(?:NIHR\s+)?Applied\s+Research\s+Collaboration\s*$RE_WEST_SOLO
    | \bARC[-\s]*$RE_WEST_SOLO
    | \bCLAHRC\s*$RE_WEST_SOLO
    | \bCollaboration\s+for\s+Leadership\s+in\s+Applied\s+Health\s+Research\s+and\s+Care\b.{0,40}$RE_WEST_SOLO
  /xmi,

  'ARC Wessex' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*Wessex\b
    | \bNIHR\s+ARC\s*Wessex\b
    | \bARC[-\s]*Wessex\b
  /xmi,

  'ARC Yorkshire and Humber' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*(?:Yorkshire\s*(?:and|&)\s*Humber)\b
    | \bNIHR\s+ARC\s*(?:Yorkshire\s*(?:and|&)\s*Humber)\b
    | \bYHARC\b
    | \bCLAHRC[-\s]*YH\b
    | \bCLAHRC\s+Yorkshire\b
  /xmi,

  'ARC East of England' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*(?:East\s+of\s+England|EoE)\b
    | \bNIHR\s+ARC\s*(?:EoE|East\s+of\s+England)\b
    | \bARC[-\s]*EoE\b
    | \bCLAHRC\s*(?:EoE|East\s+of\s+England)\b
  /xmi,

  'ARC East Midlands' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*East\s+Midlands\b
    | \bNIHR\s+ARC\s*East\s+Midlands\b
    | \bARC-EM\b
    | \(ARC-EM\)
    | \bARC[-\s]*EM\b
    | \bCLAHRC\b.{0,25}\bEast\s+Midlands\b
  /xmi,

  'ARC Greater Manchester' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*Greater\s+Manchester\b
    | \bNIHR\s+ARC[-\s]*GM\b
    | \bARC[-\s]*GM\b
    | \bARC-GM\b
  /xmi,

  'ARC Kent Surrey Sussex' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*(?:Kent\s+Surrey\s*(?:and|&)\s*Sussex)\b
    | \bNIHR\s+ARC\s*(?:KSS|Kent\s+Surrey\s*(?:and|&)\s*Sussex)\b
    | \bARC[-\s]*KSS\b
    | \bKSS\b.{0,20}\bARC\b
  /xmi,

  'ARC North East and North Cumbria' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*North\s*East\s*(?:and|&)\s*North\s+Cumbria\b
    | \bNIHR\s+ARC\s*(?:NENC)\b
    | \bNENC\b.{0,20}\bARC\b
  /xmi,

  # Strengthened North Thames (handles CLAHRC North Thames at Barts, etc.)
  'ARC North Thames' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*North\s+Thames\b
    | \bNIHR\s+ARC\s*North\s+Thames\b
    | \bCLAHRC\s+North\s+Thames\b
    | \bNorth\s+Thames\b.{0,30}\bCLAHRC\b
    | \bCLAHRC\b.{0,30}\bNorth\s+Thames\b
    | \bBarts?\b.{0,40}\bNorth\s+Thames\b
  /xmi,

  'ARC North West Coast' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*North\s*West\s+Coast\b
    | \bNIHR\s+ARC\s*(?:NWC|North\s*West\s+Coast)\b
    | \bNWC\b.{0,20}\bARC\b
    | \bCLAHRC\b.{0,25}\bNorth[\s-]*West\s+Coast\b
  /xmi,

  'ARC North West London' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*North\s*West\s+London\b
    | \bNIHR\s+ARC\s*(?:NWL|North\s*West\s+London)\b
    | \bNWL\b.{0,20}\bARC\b
    | \bCLAHRC\b.{0,25}\bNorth[\s-]*West\s+London\b
  /xmi,

  'ARC Oxford and Thames Valley' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*(?:Oxford\s*(?:and|&)\s*Thames\s+Valley|Thames\s+Valley)\b
    | \bNIHR\s+ARC\s*(?:Oxford|Thames\s+Valley)\b
  /xmi,

  'ARC South London' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*South\s+London\b
    | \bNIHR\s+ARC\s*South\s+London\b
    | \bARCSouth\s+London\b
  /xmi,

  'ARC West Midlands' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*West\s+Midlands\b
    | \bNIHR\s+ARC\s*West\s+Midlands\b
    | \bCLAHRC[-\s]*WM\b
    | \bCLAHRC\b.{0,25}\bWest\s+Midlands\b
    | \bWest\s+Midlands\b.{0,25}\bCLAHRC\b
  /xmi,

  'ARC South West Peninsula' => qr/
    \b(?:NIHR\s+)?(?:Applied\s+Research\s+Collaboration|ARC)\s*South\s*West\s+Peninsula\b
    | \bNIHR\s+ARC\s*(?:SWP|South\s*West\s+Peninsula)\b
    | \bPenARC\b
    | \bPenCLAHRC\b
    | \bCLAHRC\b.{0,25}\bSouth[\s-]*West\s+Peninsula\b
  /xmi,
);

# ----------------------------
# 5) Conservative fallback location cues (ONLY used when no explicit ARC in chunk)
# ----------------------------
my %ARC_LOC = (
  'ARC West' => qr/
    \bBristol\b(?!\s*Myer(?:s)?(?:-|\s*)Squibb\b)
    | \bUWE\b
    | \bBath\b
  /xmi,

  'ARC Wessex' => qr/\bSouthampton\b|\bHampshire\b|\bIsle\s+of\s+Wight\b/xmi,

  # IMPORTANT: no Cambridge/Cambridgeshire/Peterborough fallback (too many false positives)
  'ARC East of England' => qr/
    \bEast\s+of\s+England\b
    | \bARC[-\s]*EoE\b
    | \bEoE\b.{0,20}\bARC\b
  /xmi,

  'ARC East Midlands' => qr/\bEast\s+Midlands\b|\bLeicester(?:shire)?\b|\bNottingham(?:shire)?\b|\bDerby(?:shire)?\b|\bLincolnshire\b|\bNorthamptonshire\b/xmi,

  'ARC West Midlands' => qr/
    \bUniversity\s+of\s+Birmingham\b
    | \bKeele\b
    | \bWarwick\b
    | \bCoventry\b
    | \bStoke(?:-on-?Trent)?\b
    | \bStaffordshire\b
  /xmi,

  'ARC Yorkshire and Humber' => qr/\bYorkshire\b|\bHumber\b/xmi,

  'ARC Greater Manchester' => qr/\bGreater\s+Manchester\b/xmi,

  'ARC Kent Surrey Sussex' => qr/\bSurrey\b|\bSussex\b|\bCanterbury\b/xmi,

  # IMPORTANT: remove generic "North East" fallback (too leaky)
  'ARC North East and North Cumbria' => qr/\bNorthumberland\b|\bNorthumbria\b|\bTeesside\b|\bSunderland\b|\bCumbria\b|\bNewcastle\s+upon\s+Tyne\b/xmi,

  'ARC North West Coast' => qr/\bNorth\s*West\s+Coast\b|\bLiverpool\b|\bCheshire\b/xmi,

  'ARC North West London' => qr/\bNorth\s*West\s+London\b|\bNWL\b/xmi,

  # Optional light fallback for North Thames; keep minimal
  'ARC North Thames' => qr/\bNorth\s+Thames\b/xmi,
);

# ----------------------------
# Debug collector
# ----------------------------
my $csv_out_debug = Text::CSV->new({ binary => 1, eol => "\n" });
open my $fh_debug, ">:encoding(UTF-8)", $OUT_DEBUG or die "Cannot write $OUT_DEBUG: $!";

$csv_out_debug->print($fh_debug, [qw/
  paper_id field chunk_index chunk_len
  has_anchor has_general has_exclusion kent_surname
  phase arc_name rule
  anchor_match explicit_match loc_match
  chunk_preview
/]);

sub debug_row {
  my (%d) = @_;
  $csv_out_debug->print($fh_debug, [
    $d{paper_id}, $d{field}, $d{chunk_index}, $d{chunk_len},
    $d{has_anchor}, $d{has_general}, $d{has_exclusion}, $d{kent_surname},
    $d{phase}, $d{arc_name}, $d{rule},
    $d{anchor_match}, $d{explicit_match}, $d{loc_match},
    $d{chunk_preview}
  ]);
}

# ----------------------------
# Resolver: prefer explicit over fallback; clean up known spillovers
# ----------------------------
sub resolve_west_vs_wm {
  my ($hit_reason_ref) = @_;
  my %hr = %{$hit_reason_ref};

  my $has_w  = exists $hr{'ARC West'};
  my $has_wm = exists $hr{'ARC West Midlands'};
  return %hr unless ($has_w && $has_wm);

  my $w_reason  = $hr{'ARC West'} // "";
  my $wm_reason = $hr{'ARC West Midlands'} // "";

  if ($wm_reason =~ /:explicit/ && $w_reason =~ /fallback/) {
    delete $hr{'ARC West'};
    return %hr;
  }
  if ($w_reason =~ /:explicit/ && $wm_reason =~ /fallback/) {
    delete $hr{'ARC West Midlands'};
    return %hr;
  }

  return %hr;
}

sub resolve_competing_fallbacks {
  my ($hit_reason_ref, $fu_text, $addr_text) = @_;
  my %hr = %{$hit_reason_ref};

  my $txt = ($fu_text // "") . " " . ($addr_text // "");

  # If any explicit hit exists, be stricter about leaving weak spillover fallbacks
  my $has_any_explicit = 0;
  for my $k (keys %hr) {
    if (($hr{$k} // "") =~ /:explicit/) { $has_any_explicit = 1; last; }
  }

  # If West is only fallback and text clearly refers to North West / South West / West Midlands, drop West fallback
  if (exists $hr{'ARC West'} && ($hr{'ARC West'} // "") =~ /fallback/) {
    if ($txt =~ /\b(?:North|South)[\s-]+West\b/i || $txt =~ /\bWest\s+Midlands\b/i) {
      delete $hr{'ARC West'};
    }
  }

  # If EM + EoE are both present and one is fallback spillover, decide using strong cues
  if (exists $hr{'ARC East Midlands'} && exists $hr{'ARC East of England'}) {
    my $em  = $hr{'ARC East Midlands'} // "";
    my $eoe = $hr{'ARC East of England'} // "";

    if ($em =~ /fallback/ && $eoe =~ /fallback/) {
      if ($txt =~ /\bEast\s+Midlands\b/i || $txt =~ /\bNottingham(?:shire)?\b/i || $txt =~ /\bLeicester(?:shire)?\b/i || $txt =~ /\bDerby(?:shire)?\b/i) {
        delete $hr{'ARC East of England'};
      } elsif ($txt =~ /\bEast\s+of\s+England\b/i) {
        delete $hr{'ARC East Midlands'};
      }
    }

    # If one is explicit, drop the other if it is only fallback
    if ($em =~ /:explicit/ && $eoe =~ /fallback/) { delete $hr{'ARC East of England'}; }
    if ($eoe =~ /:explicit/ && $em =~ /fallback/) { delete $hr{'ARC East Midlands'}; }
  }

  # If North Thames exists, drop EoE/NENC spillover fallbacks that often come from Cambridge/London mentions
  if (exists $hr{'ARC North Thames'}) {
    if (exists $hr{'ARC East of England'} && ($hr{'ARC East of England'} // "") =~ /fallback/) {
      delete $hr{'ARC East of England'};
    }
    if (exists $hr{'ARC North East and North Cumbria'} && ($hr{'ARC North East and North Cumbria'} // "") =~ /fallback/) {
      delete $hr{'ARC North East and North Cumbria'};
    }
  }

  # If there are explicit hits, optionally drop pure fallback hits (conservative tightening)
  if ($has_any_explicit) {
    for my $k (keys %hr) {
      if (($hr{$k} // "") =~ /fallback/) {
        # Keep fallback only if it is the only evidence for that ARC and does not conflict with obvious region strings
        # (This is intentionally light; extend if needed.)
      }
    }
  }

  return %hr;
}

# ----------------------------
# Matching function: explicit-first per chunk, with debug
# ----------------------------
sub match_chunk_arcs {
  my (%args) = @_;
  my $chunk      = $args{chunk} // "";
  my $field      = $args{field} // "unknown";
  my $paper_id   = $args{paper_id} // "NOID";
  my $chunk_i    = $args{chunk_index} // 0;
  my $has_gen    = $args{has_general} // 0;
  my $has_excl   = $args{has_exclusion} // 0;

  my %hits;

  my $kent_surname = has_kent_surname($chunk) ? 1 : 0;
  my $has_anchor   = ($chunk =~ $RE_INFRA_ANCHOR) ? 1 : 0;

  debug_row(
    paper_id => $paper_id,
    field => $field,
    chunk_index => $chunk_i,
    chunk_len => length($chunk),
    has_anchor => $has_anchor,
    has_general => $has_gen,
    has_exclusion => $has_excl,
    kent_surname => $kent_surname,
    phase => "chunk",
    arc_name => "",
    rule => "",
    anchor_match => ($has_anchor ? "Y" : "N"),
    explicit_match => "",
    loc_match => "",
    chunk_preview => shorten($chunk, 300)
  );

  return %hits unless $has_anchor;

  # 1) Explicit-first
  my @explicit;
  for my $arc (sort keys %ARC_EXPLICIT) {
    if ($chunk =~ /($ARC_EXPLICIT{$arc})/) {
      my $m = $1;
      push @explicit, $arc;

      debug_row(
        paper_id => $paper_id,
        field => $field,
        chunk_index => $chunk_i,
        chunk_len => length($chunk),
        has_anchor => $has_anchor,
        has_general => $has_gen,
        has_exclusion => $has_excl,
        kent_surname => $kent_surname,
        phase => "explicit",
        arc_name => $arc,
        rule => "explicit_lock",
        anchor_match => "Y",
        explicit_match => shorten($m, 120),
        loc_match => "",
        chunk_preview => shorten($chunk, 240)
      );
    }
  }

  if (@explicit) {
    for my $arc (@explicit) {
      $hits{$arc} = $field . ":explicit";
    }

    debug_row(
      paper_id => $paper_id,
      field => $field,
      chunk_index => $chunk_i,
      chunk_len => length($chunk),
      has_anchor => $has_anchor,
      has_general => $has_gen,
      has_exclusion => $has_excl,
      kent_surname => $kent_surname,
      phase => "fallback",
      arc_name => "",
      rule => "skipped_due_to_explicit_lock",
      anchor_match => "Y",
      explicit_match => join("|", @explicit),
      loc_match => "",
      chunk_preview => shorten($chunk, 180)
    );

    return %hits;
  }

  # 2) Fallback: conservative loc cues
  for my $arc (sort keys %ARC_LOC) {

    # Block ARC West fallback if chunk is talking about North West / South West / West Midlands etc.
    if ($arc eq 'ARC West') {
      if ($chunk =~ /\b(?:North|South)[\s-]+West\b/i || $chunk =~ /\bWest\s+Midlands\b/i) {
        debug_row(
          paper_id => $paper_id,
          field => $field,
          chunk_index => $chunk_i,
          chunk_len => length($chunk),
          has_anchor => $has_anchor,
          has_general => $has_gen,
          has_exclusion => $has_excl,
          kent_surname => $kent_surname,
          phase => "fallback",
          arc_name => $arc,
          rule => "blocked_west_due_to_compound_west",
          anchor_match => "Y",
          explicit_match => "",
          loc_match => "",
          chunk_preview => shorten($chunk, 180)
        );
        next;
      }
    }

    # Avoid KSS fallback if Kent surname present
    if ($arc eq 'ARC Kent Surrey Sussex' && $kent_surname) {
      debug_row(
        paper_id => $paper_id,
        field => $field,
        chunk_index => $chunk_i,
        chunk_len => length($chunk),
        has_anchor => $has_anchor,
        has_general => $has_gen,
        has_exclusion => $has_excl,
        kent_surname => $kent_surname,
        phase => "fallback",
        arc_name => $arc,
        rule => "blocked_kent_surname",
        anchor_match => "Y",
        explicit_match => "",
        loc_match => "",
        chunk_preview => shorten($chunk, 180)
      );
      next;
    }

    # Block WM fallback when Birmingham mention is just Birmingham BRC (common false positive)
    if ($arc eq 'ARC West Midlands' && has_birmingham_brc_only($chunk)) {
      debug_row(
        paper_id => $paper_id,
        field => $field,
        chunk_index => $chunk_i,
        chunk_len => length($chunk),
        has_anchor => $has_anchor,
        has_general => $has_gen,
        has_exclusion => $has_excl,
        kent_surname => $kent_surname,
        phase => "fallback",
        arc_name => $arc,
        rule => "blocked_birmingham_brc",
        anchor_match => "Y",
        explicit_match => "",
        loc_match => "Birmingham Biomedical Research Centre",
        chunk_preview => shorten($chunk, 180)
      );
      next;
    }

    my $loc = $ARC_LOC{$arc};
    if ($chunk =~ /($loc)/) {
      my $m = $1;
      $hits{$arc} = $field . ":fallback_loc";

      debug_row(
        paper_id => $paper_id,
        field => $field,
        chunk_index => $chunk_i,
        chunk_len => length($chunk),
        has_anchor => $has_anchor,
        has_general => $has_gen,
        has_exclusion => $has_excl,
        kent_surname => $kent_surname,
        phase => "fallback",
        arc_name => $arc,
        rule => "fallback_loc",
        anchor_match => "Y",
        explicit_match => "",
        loc_match => shorten($m, 120),
        chunk_preview => shorten($chunk, 240)
      );
    }
  }

  return %hits;
}

# ----------------------------
# CSV IO
# ----------------------------
open my $fh, "<:raw", $INPUT_CSV or die "Cannot open $INPUT_CSV: $!";

my $csv_in = Text::CSV->new({
  binary => 1,
  decode_utf8 => 1,
  allow_loose_quotes  => 1,
  allow_loose_escapes => 1,
  auto_diag => 1,
});

my $header = $csv_in->getline($fh);
$header->[0] =~ s/^\x{FEFF}//;

die "Empty input file\n" unless $header && @$header;

my %col_index;
for my $i (0..$#$header) {
  $col_index{$header->[$i]} = $i;
}

sub row_to_hash {
  my ($aref) = @_;
  my %h;
  for my $k (keys %col_index) {
    $h{$k} = $aref->[ $col_index{$k} ];
  }
  return \%h;
}

my $csv_out_flags = Text::CSV->new({ binary => 1, eol => "\n" });
my $csv_out_long  = Text::CSV->new({ binary => 1, eol => "\n" });

open my $fh_flags, ">:encoding(UTF-8)", $OUT_FLAGS or die "Cannot write $OUT_FLAGS: $!";
open my $fh_long,  ">:encoding(UTF-8)", $OUT_LONG  or die "Cannot write $OUT_LONG: $!";

$csv_out_flags->print($fh_flags, [qw/
  paper_id keep exclude_reason
  has_general n_specific specific_list specific_reasons
  false_hit_addr false_hit_fu
/]);

$csv_out_long->print($fh_long, [qw/paper_id infra_name match_field/]);

# ----------------------------
# Main loop
# ----------------------------
my $n = 0;
my $kept = 0;
my $excluded = 0;

while (my $row_aref = $csv_in->getline($fh)) {
  $n++;

  my $row = row_to_hash($row_aref);
  my $paper_id = get_paper_id($row);

  my $addr = norm_text($row->{$COL_ADDR});
  my $fu   = norm_text($row->{$COL_FU});

  # Exclusion flags
  my $false_addr = ($addr =~ $RE_EXCL_ADDR) ? 1 : 0;
  my $false_fu   = ($fu   =~ $RE_EXCL_FU)   ? 1 : 0;

  # General infra signal
  my $has_general_addr = ($addr =~ $RE_GENERAL) ? 1 : 0;
  my $has_general_fu   = ($fu   =~ $RE_GENERAL) ? 1 : 0;
  my $has_general      = ($has_general_addr || $has_general_fu) ? 1 : 0;

  my $exclude = 0;
  my $exclude_reason = "";

  if (!$has_general) {
    $exclude = 1;
    $exclude_reason = "no_infra_signal";
  }

  my %hit_reason;

  if (!$exclude) {

    if ($has_general_addr) {
      my @chunks = split_chunks($addr);
      for (my $i = 0; $i < @chunks; $i++) {
        my %hits = match_chunk_arcs(
          chunk => $chunks[$i],
          field => "addr",
          paper_id => $paper_id,
          chunk_index => $i,
          has_general => $has_general,
          has_exclusion => $false_addr
        );
        @hit_reason{keys %hits} = values %hits if %hits;
      }
    }

    if ($has_general_fu) {
      my @chunks = split_chunks($fu);
      for (my $i = 0; $i < @chunks; $i++) {
        my %hits = match_chunk_arcs(
          chunk => $chunks[$i],
          field => "fu",
          paper_id => $paper_id,
          chunk_index => $i,
          has_general => $has_general,
          has_exclusion => $false_fu
        );
        @hit_reason{keys %hits} = values %hits if %hits;
      }
    }
  }

  # Post-match cleanup / disambiguation
  %hit_reason = resolve_west_vs_wm(\%hit_reason);
  %hit_reason = resolve_competing_fallbacks(\%hit_reason, $fu, $addr);

  my @specific = sort keys %hit_reason;
  my $n_specific = scalar @specific;
  my $specific_list = join(" | ", @specific);
  my $specific_reasons = join(" | ", map { $_ . "=" . $hit_reason{$_} } @specific);

  my $keep = $exclude ? 0 : 1;
  $kept++ if $keep;
  $excluded++ if $exclude;

  $csv_out_flags->print($fh_flags, [
    $paper_id,
    $keep,
    $exclude_reason,
    $has_general,
    $n_specific,
    $specific_list,
    $specific_reasons,
    $false_addr,
    $false_fu
  ]);

  if ($keep && $n_specific > 0) {
    for my $arc (@specific) {
      $csv_out_long->print($fh_long, [
        $paper_id,
        $arc,
        $hit_reason{$arc}
      ]);
    }
  }
}

close $fh;
close $fh_flags;
close $fh_long;
close $fh_debug;

print "Processed $n rows\n";
print "Kept:     $kept\n";
print "Excluded: $excluded\n";
print "Wrote $OUT_FLAGS, $OUT_LONG, $OUT_DEBUG\n";
