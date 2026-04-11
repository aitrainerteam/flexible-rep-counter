nd-to-End Data Flow: Client → Cloud → Client
1. User UID Processing
Client side (auxiliary_agents.py):

_resolve_vm_rep_user_uid() resolves in priority order: VM_REP_USER_UID_OVERRIDE → WORKOUT_PLAN_USER_UID_OVERRIDE → human_identity
The resolved UID is stored in _yolo_rep_runtime["user_uid"] via _set_yolo_rep_runtime(user_uid=resolved)
On every /predict call, it's snapshotted and sent as ?user_uid=<value>
Cloud side (yolo-deploy/main.py:345):

uid = user_uid or form_user_uid
The UID is the session key into RepSessionManager.sessions dict (in-memory, keyed by string)
Same UID = same session = cumulative reps continue from where they left off
2. rep_enabled Flag
Client (auxiliary_agents.py:1156-1158):

Sent as rep_enabled=true only when is_execution_phase == True
During explanation/rest phases → rep_enabled=false
Cloud (yolo-deploy/main.py:347,358):

is_rep = True only if rep_enabled is "1", "true", or "yes"
Rep counting block (line 358) only runs when is_rep and uid and not is_exercise_ended
When rep_enabled=false: no rep_counter in response, but the session stays alive in memory with its accumulated reps
This is the critical behavior: the cloud session persists even when rep_enabled=false. It just doesn't process new frames. The reps don't reset — they freeze.

3. exercise_ended Flag and Session Reset
Client (auxiliary_agents.py:3855-3856):

exercise_ended_pending=True is set only when is_last_set_of_exercise is True (current set number == total sets)
It's sent as ?exercise_ended=true on the next /predict call
It's cleared only after the cloud confirms with a rep_session_reset dict in the response
Cloud (yolo-deploy/main.py:351-356):

When exercise_ended=true: calls rep_session_manager.reset(uid) which deletes the session entirely
Returns {"rep_session_reset": {"requested": True, "had_session": True/False}}
On the next request, get_or_create() creates a brand-new RepCounterSession starting at 0 reps
4. Set Handling (Multi-Set Exercises)
Client (auxiliary_agents.py:4120-4121):

On start_set_explanation: vm_prev_reps = vm_last_seen_reps
This snapshots the current session total so the subtraction set_reps = session_reps - vm_prev_reps starts the new set at 0
Cloud: The cloud has no concept of sets. It just accumulates reps in a single session. The per-set math is entirely client-side.

5. The Rep Subtraction Math
set_reps = max(0, session_reps - vm_prev_reps)
session_reps = rep_counter.reps from the cloud (cumulative for the session)
vm_prev_reps = baseline snapshot (0 at exercise start, vm_last_seen_reps at set boundaries)