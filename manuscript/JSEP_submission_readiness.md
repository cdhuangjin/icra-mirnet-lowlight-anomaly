# Journal of Software: Evolution and Process — Submission Readiness Audit

Source reviewed: `industrial_lowlight_inspection.tex`; requirements source: the
provided Wiley JSEP author-guideline file.

## Ready or substantially ready

| Requirement | Evidence | Status |
|---|---|---|
| Editable manuscript | LaTeX source exists | Ready |
| Peer-review PDF | `manuscript/build/industrial_lowlight_inspection.pdf` exists, but it predates the author update | Recompile before upload |
| Main sections | Abstract, introduction, method, experiments, conclusion | Ready |
| Abstract length | Approximately 170 words; no citations | Ready |
| Keywords | Five supplied | Ready |
| References | Consistent BibTeX bibliography | Ready |
| Evidence boundary | Synthetic-low-light and detector-proxy limitations disclosed | Ready |

## Submission blockers requiring author-supplied information

| Requirement | Current state | Required action |
|---|---|---|
| Author details and affiliations | Three authors, affiliations, emails, and corresponding author are recorded in the source and title page | Telephone number for corresponding author remains needed for submission metadata. |
| ORCID | Jin Huang's ORCID is recorded | Coauthor ORCIDs are optional and have not been supplied. |
| Title page | `JSEP_title_page.md` contains authors and declarations | Add the corresponding-author telephone and final repository URL. |
| Data availability statement | Dataset provenance and a repository placeholder are prepared | Replace placeholder with the final repository URL, release commit/tag, and license. |
| Funding statement | No-funding statement provided | Ready. |
| Conflict-of-interest statement | No-conflict statement provided | Ready. |
| Ethics/consent statements | Not present | Confirm whether not applicable; do not invent an approval number. |
| Permissions | Not stated | Confirm all figures/tables are original or identify permissions. |

## Format changes needed before upload

- The manuscript currently uses `IEEEtran`, not Wiley's NJD template. JSEP accepts
  free-format initial submission, so conversion is not a hard initial-submission
  blocker, but a Wiley/NJD conversion should be made before revision-stage upload.
- The guideline requests tables on separate pages after references and figures as
  separate high-resolution files. The current two tables are embedded in the text
  and no standalone figures are present.
- JSEP requires a graphical table-of-contents figure plus up to 80 words of text;
  neither is currently provided.
- For a LaTeX upload, retain the `.tex`, `.bib`, supporting files, and a matching
  PDF; designate the `.tex` and PDF as the appropriate main-document file types.

## Safe next action

Create the repository named `icra-mirnet-industrial-lowlight` and provide its URL.
Then supply the corresponding-author telephone and confirm the ethics/consent and
permissions statements. A full TeX Live installation (including BibTeX/latexmk)
is also needed to regenerate the author-version peer-review PDF; the bundled
Tectonic smoke test works, but it cannot compile this bibliography project through
the current submission build workflow.
