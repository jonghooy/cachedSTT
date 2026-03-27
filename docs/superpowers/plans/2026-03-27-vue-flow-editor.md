# Vue Flow Visual Editor — Plan 4

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Add scenario visual editor to Knowledge Service frontend using Vue Flow. 3-panel layout: scenario list + AI chat | graph canvas | node properties.

**Architecture:** New Vue 3 view (ScenariosView.vue) with Vue Flow graph editor. Custom node components for each of 9 node types. Integrates with existing /api/scenarios CRUD + /generate/refine endpoints.

**Tech Stack:** Vue 3, Vue Flow (@vue-flow/core), existing Axios API client, existing router

**Working directory:** `/home/jonghooy/work/knowledge-service`

---

## File Structure

```
frontend/src/
├── views/
│   └── ScenariosView.vue      # NEW: main 3-panel layout
├── components/
│   └── scenarios/
│       ├── ScenarioGraph.vue   # NEW: Vue Flow canvas
│       ├── NodeProperties.vue  # NEW: right panel property editor
│       ├── AiChat.vue          # NEW: left panel AI chat
│       ├── ScenarioList.vue    # NEW: left panel scenario list
│       └── nodes/              # NEW: custom node components
│           ├── SpeakNode.vue
│           ├── SlotNode.vue
│           ├── ConditionNode.vue
│           ├── ConfirmNode.vue
│           ├── ApiCallNode.vue
│           ├── EndNode.vue
│           ├── TransferNode.vue
│           ├── LlmNode.vue
│           └── RagNode.vue
├── router.js                   # MODIFY: add /scenarios route
└── api/
    └── client.js               # EXISTING (no changes)
```

---

### Task 1: Install Vue Flow + Create Router Entry
### Task 2: ScenarioList + AiChat (Left Panel)
### Task 3: Custom Node Components (9 types)
### Task 4: ScenarioGraph (Vue Flow Canvas)
### Task 5: NodeProperties (Right Panel Editor)
### Task 6: ScenariosView (Main 3-Panel Layout)

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Vue Flow setup + router | package.json, router.js |
| 2 | Left panel (list + AI chat) | ScenarioList.vue, AiChat.vue |
| 3 | Custom nodes (9 types) | nodes/*.vue |
| 4 | Graph canvas | ScenarioGraph.vue |
| 5 | Property editor | NodeProperties.vue |
| 6 | Main view | ScenariosView.vue |
