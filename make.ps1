<#
.SYNOPSIS
    The Makefile's targets, for Windows participants who do not have make.

.EXAMPLE
    .\make.ps1 doctor
    .\make.ps1 ask -Q "How do I renew my commercial licence?"
    .\make.ps1 extract-corpus -Route vllm
    .\make.ps1 replay -Label after -Cache -Semantic -Routing

.NOTES
    Same target names as the Makefile, same behaviour. If you have make (Git Bash,
    WSL, or scoop install make), use that instead — this exists so that not having
    it never costs anybody class time.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Target = 'help',
    [string]$Q = 'How do I renew my commercial licence?',
    [string]$Route = '',
    [string]$Label = 'run',
    [string]$Prompt = '',
    [int]$Limit = 200,
    [switch]$Cache,
    [switch]$Semantic,
    [switch]$Routing,
    [switch]$Cascade
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $root 'src'

function Invoke-Py { param([string[]]$Arguments) & $python @Arguments; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function RouteFlag { if ($Route) { @('--route', $Route) } else { @() } }

switch ($Target) {
    'help' {
        @'
  install          create .venv and install requirements.lock
  doctor           check python, config, Arabic rendering and every route
  gateway          run the course gateway on port 8080 (Ctrl-C to stop)
  stats            gateway counters
  drill-429        a 429 storm on the primary model for three minutes
  drill-outage     the primary model 529s; the fallback hop should serve
  drill-off        end any running drill
  ask              one question:  .\make.ps1 ask -Q "..." [-Route vllm]
  chat             a windowed conversation
  stream           stream one answer and report TTFT
  test             pytest
  lint             ruff
  schema-check     strict-mode subset check for every output contract
  bench            20 bilingual prompts against every route
  token-report     the Arabic token premium per tokenizer
  extract-corpus   schema-pass rate over the 50-case corpus
  extract-audit    the same, plus the invented-field audit
  tool-smoke       the scripted conversation and its expected tool calls
  guard-eval       block rate AND false-positive rate
  leak-attack      five system-prompt extraction attempts
  golden           rebuild the golden set
  eval             the golden set through the real pipeline
  calibrate        judge vs the 40 human labels, both rubrics
  gate             compare the last run against the baseline
  baseline         promote the last run to the baseline (a governed act)
  eval-report      regenerate EVALUATION_REPORT.md
  replay           meter a replay:  .\make.ps1 replay -Label after -Cache -Routing
  eval-cache       the semantic cache near-miss suite
  breakeven        self-host vs commercial
  corpora          regenerate every corpus
'@
    }
    'install' {
        & python -m venv (Join-Path $root '.venv')
        Invoke-Py @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Py @('-m', 'pip', 'install', '-r', (Join-Path $root 'requirements.lock'))
    }
    'doctor' { Invoke-Py @('-m', 'retail_support.cli', 'doctor') }
    'gateway' {
        if (-not $env:MOCKGW_SPEED) { $env:MOCKGW_SPEED = '0.2' }
        Push-Location (Join-Path $root 'infra\mockgw')
        try { & $python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 } finally { Pop-Location }
    }
    'stats' { (Invoke-WebRequest -Uri 'http://127.0.0.1:8080/admin/stats' -UseBasicParsing).Content }
    'drill-429' {
        Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/admin/fault' -ContentType 'application/json' `
            -Body '{"mode":"rate_limit","seconds":180,"model":"course-flagship","retry_after":2}' | ConvertTo-Json -Depth 5
    }
    'drill-outage' {
        Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/admin/fault' -ContentType 'application/json' `
            -Body '{"mode":"overload","seconds":300,"model":"course-flagship"}' | ConvertTo-Json -Depth 5
    }
    'drill-off' {
        Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/admin/fault' -ContentType 'application/json' `
            -Body '{"mode":"off"}' | ConvertTo-Json -Depth 5
    }
    'ask' { Invoke-Py (@('-m', 'retail_support.cli') + (RouteFlag) + @('ask', $Q)) }
    'chat' { Invoke-Py (@('-m', 'retail_support.cli') + (RouteFlag) + @('chat')) }
    'stream' { Invoke-Py (@('-m', 'retail_support.cli') + (RouteFlag) + @('stream', $Q)) }
    'test' { Invoke-Py @('-m', 'pytest') }
    'lint' { Invoke-Py @('-m', 'ruff', 'check', 'src', 'tests', 'scripts', 'eval', 'infra') }
    'schema-check' { Invoke-Py @('scripts/schema_check.py') }
    'bench' { Invoke-Py @('scripts/bench_providers.py') }
    'token-report' { Invoke-Py @('scripts/token_report.py') }
    'extract-corpus' { Invoke-Py (@('scripts/extract_corpus.py') + (RouteFlag)) }
    'extract-audit' { Invoke-Py (@('scripts/extract_corpus.py', '--audit') + (RouteFlag)) }
    'tool-smoke' { Invoke-Py @('scripts/tool_smoke.py') }
    'guard-eval' { Invoke-Py @('scripts/guard_eval.py') }
    'leak-attack' { Invoke-Py @('scripts/leak_attack.py') }
    'golden' { Invoke-Py @('eval/build_golden.py') }
    'eval' { Invoke-Py (@('eval/harness.py', '--label', $Label) + (RouteFlag)) }
    'calibrate' {
        Invoke-Py @('eval/build_human_labels.py')
        & $python eval/calibrate_judge.py --rubric groundedness.v1.md
        Invoke-Py @('eval/calibrate_judge.py', '--rubric', 'groundedness.v2.md')
    }
    'gate' { Invoke-Py @('eval/gate.py', "eval/out/eval_$Label.json", '--baseline', 'eval/baseline.json') }
    'baseline' { Invoke-Py @('eval/promote_baseline.py', $Label) }
    'eval-report' { Invoke-Py @('eval/report.py') }
    'replay' {
        $args = @('scripts/replay.py', '--label', $Label, '--limit', "$Limit") + (RouteFlag)
        if ($Cache) { $args += '--cache' }
        if ($Semantic) { $args += '--semantic' }
        if ($Routing) { $args += '--routing' }
        if ($Cascade) { $args += '--cascade' }
        if ($Prompt) { $args += @('--prompt', $Prompt) }
        Invoke-Py $args
    }
    'eval-cache' { Invoke-Py @('scripts/eval_cache.py') }
    'breakeven' { Invoke-Py @('scripts/breakeven.py') }
    'corpora' { Invoke-Py @('scripts/generate_corpora.py') }
    default { Write-Error "unknown target '$Target' — run .\make.ps1 help" }
}
