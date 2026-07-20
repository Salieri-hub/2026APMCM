# TODO

## Completed

- [x] Migrate default backbone from `EfficientNet-B1` to `EfficientNet-B2`
- [x] Update default image size from `240` to `256`
- [x] Add local pretrained weight loading support for `B2`
- [x] Keep compatibility with historical `B0/B1` checkpoints
- [x] Change output layout to:
  - `outputs/weights/<experiment_name>/`
  - `outputs/results/<experiment_name>/`
- [x] Create `B2` batch scripts for `50` formal experiments
- [x] Update all project markdown files to the `B2` baseline

## In Progress

- [ ] Run the `50` formal `B2` experiments
- [ ] Summarize `B2` vs historical `B0` comparison
- [ ] Update Word reports after `B2` results are available

## Optional Next Steps

- [ ] Tune cascade trigger threshold after the `B2` run finishes
- [ ] Add a consolidated `B0` vs `B2` comparison docx
- [ ] Add expert-only result summary for the `40` expert training runs
