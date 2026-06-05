"""Quick local test of the remote PBS backend (path B) — no MCP client needed.

Exercises hpc_server.submit_md_job + check_job with APW_REMOTE_CMD set, so the
actual plink->qsub / plink->qstat code path runs. Env must be set BEFORE running
(the module reads it at import). PowerShell:

  $env:APW_REMOTE_CMD       = 'plink -batch -P 60026 <user>@<pbs-host>'
  $env:APW_REMOTE_WORKDIR   = '/scratch/<user>/gmx_smoke'   # reuse smoke.tpr
  $env:APW_PBS_QUEUE        = 'nvidiaq'
  $env:APW_PBS_NGPUS        = '1'
  $env:APW_PBS_GMX_BIN      = 'gmx_mpi'
  $env:APW_PBS_EXTRA_MODULES= 'CUDA/12.4 OPENMPI/4.1.6.GCC5.8'
  python examples/test_remote_submit.py smoke

Requires <deffnm>.tpr already present in APW_REMOTE_WORKDIR on the cluster.
"""
import sys
import time

sys.path.insert(0, "mcp_servers")
import hpc_server as h  # noqa: E402

deffnm = sys.argv[1] if len(sys.argv) > 1 else "smoke"
print("REMOTE_CMD   =", h.REMOTE_CMD or "(empty -> local)")
print("REMOTE_WORK  =", h.REMOTE_WORKDIR)
print("queue/ngpus  =", h.PBS_QUEUE, h.PBS_NGPUS)

res = h.submit_md_job(deffnm=deffnm, backend="pbs")
print("submit ->", res)
if not res.get("ok"):
    sys.exit(1)

job_id = res["job_id"]
for i in range(20):
    st = h.check_job(job_id)
    print(f"poll {i:02d} ->", st)
    if st.get("state") in ("FINISHED", "GONE", "COMPLETED_OR_GONE"):
        break
    time.sleep(15)
