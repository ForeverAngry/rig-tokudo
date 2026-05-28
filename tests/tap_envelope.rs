//! Feature test for the `tap` integration: tokudo keeps its own event
//! payload, but emits rig_tap-compatible scalar fields on the same tracing
//! event so collectors can route both streams uniformly.

#![cfg(feature = "tap")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::{Arc, Mutex};

use rig_tokudo::RouterChoice;
use rig_tokudo::observe::{self, TokudoEvent};
use tracing::field::{Field, Visit};
use tracing::subscriber::with_default;
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

#[derive(Default)]
struct CapturedEvent {
    event_json: Option<String>,
    rig_tap_kind: Option<String>,
    rig_tap_version: Option<u64>,
}

#[derive(Clone, Default)]
struct CapturingLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

struct CaptureVisitor<'a> {
    captured: &'a mut CapturedEvent,
}

impl Visit for CaptureVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "event" => self.captured.event_json = Some(value.to_string()),
            "rig_tap.kind" => self.captured.rig_tap_kind = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "rig_tap.version" {
            self.captured.rig_tap_version = Some(value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        match field.name() {
            "event" => {
                let trimmed = rendered.trim_matches('"').replace("\\\"", "\"");
                self.captured.event_json = Some(trimmed);
            }
            "rig_tap.kind" => {
                self.captured.rig_tap_kind = Some(rendered.trim_matches('"').to_string());
            }
            "rig_tap.version" => {
                if let Ok(version) = rendered.parse::<u64>() {
                    self.captured.rig_tap_version = Some(version);
                }
            }
            _ => {}
        }
    }
}

impl<S: Subscriber> Layer<S> for CapturingLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != observe::EVENT_TARGET {
            return;
        }
        let mut captured = CapturedEvent::default();
        let mut visitor = CaptureVisitor {
            captured: &mut captured,
        };
        event.record(&mut visitor);
        if let Ok(mut guard) = self.events.lock() {
            guard.push(captured);
        }
    }
}

#[test]
fn tap_feature_adds_aligned_scalar_fields() {
    let capture = CapturingLayer::default();
    let events = capture.events.clone();
    let subscriber = Registry::default().with(capture);

    with_default(subscriber, || {
        observe::emit(&TokudoEvent::RouteDecision {
            choice: RouterChoice::Strong,
        });
    });

    let guard = events.lock().unwrap();
    assert_eq!(guard.len(), 1);
    let event = &guard[0];
    assert_eq!(event.rig_tap_kind.as_deref(), Some("route.decision"));
    assert_eq!(event.rig_tap_version, Some(rig_tap::SCHEMA_VERSION as u64));

    let json = event.event_json.as_ref().expect("event JSON captured");
    let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(parsed["kind"], "route.decision");
    assert_eq!(parsed["choice"], "strong");
}
