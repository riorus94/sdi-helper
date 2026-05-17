"""Outer loop: iterate queries × sources × candidates until quota fills or queries exhaust."""

import logging
from dataclasses import dataclass

from sdi_helper.application.dto.process_result import ProcessOutcome

log = logging.getLogger(__name__)
from sdi_helper.application.dto.scrape_report import ScrapeReport
from sdi_helper.application.ports.image_source import ImageSource
from sdi_helper.application.ports.quota_repository import QuotaRepository
from sdi_helper.application.use_cases.process_candidate_image import ProcessCandidateImage


@dataclass
class ScrapeUntilQuotaFilled:
    sources: list[ImageSource]
    process: ProcessCandidateImage
    quota_repository: QuotaRepository
    queries: list[str]
    max_results_per_query: int = 200
    checkpoint_every: int = 50

    def execute(self) -> ScrapeReport:
        report = ScrapeReport()
        processed_since_checkpoint = 0

        try:
            for query in self.queries:
                if self.process.quota.all_full():
                    break

                for source in self.sources:
                    if self.process.quota.all_full():
                        break

                    try:
                        candidates = source.search(query, self.max_results_per_query)
                    except Exception as exc:
                        log.warning("source=%s query=%r search failed", source.name, query, exc_info=True)
                        continue

                    for candidate in candidates:
                        if self.process.quota.all_full():
                            break
                        try:
                            result = self.process.execute(candidate)
                        except Exception as exc:
                            log.error("failed to process candidate url=%s", candidate.image_url, exc_info=True)
                            report.record_reject(ProcessOutcome.REJECTED_DOWNLOAD, source.name)
                            continue

                        if result.outcome == ProcessOutcome.ACCEPTED and result.image is not None:
                            report.record_accept(result.image.view)
                            log.info(
                                "ACCEPT view=%s conf=%.2f split=%s total=%d",
                                result.image.view.value,
                                result.image.view_confidence,
                                result.image.split.value,
                                report.total_accepted(),
                            )
                        else:
                            report.record_reject(result.outcome, source.name)
                            if result.reason_detail:
                                log.debug("REJECT outcome=%s detail=%s", result.outcome.value, result.reason_detail)
                            else:
                                log.debug("REJECT outcome=%s", result.outcome.value)

                        processed_since_checkpoint += 1
                        if processed_since_checkpoint >= self.checkpoint_every:
                            self.quota_repository.save(self.process.quota)
                            self.process.dedup.flush()
                            processed_since_checkpoint = 0
        finally:
            self.quota_repository.save(self.process.quota)
            try:
                self.process.dedup.flush()
            except Exception as exc:
                log.warning("dedup flush failed", exc_info=True)
            for source in self.sources:
                try:
                    source.close()
                except Exception:
                    pass

        return report
