export class HelixError extends Error {
  readonly helixStatus: number;
  readonly helixErrorJson: string;
  readonly param?: string;
  readonly code?: string;

  constructor(message: string, status: number, errorJson: string) {
    super(message);
    this.name = "HelixError";
    this.helixStatus = status;
    this.helixErrorJson = errorJson;
    try {
      const parsed = JSON.parse(errorJson);
      this.param = parsed?.error?.param ?? undefined;
      this.code = parsed?.error?.code ?? undefined;
    } catch {}
  }
}

export class HelixValidationError extends HelixError { override name = "HelixValidationError"; }
export class HelixModelNotFoundError extends HelixError { override name = "HelixModelNotFoundError"; }
export class HelixModelLoadFailedError extends HelixError { override name = "HelixModelLoadFailedError"; }
export class HelixOomError extends HelixError { override name = "HelixOomError"; }
export class HelixCancelledError extends HelixError { override name = "HelixCancelledError"; }
export class HelixContextFullError extends HelixError { override name = "HelixContextFullError"; }
export class HelixUnsupportedFeatureError extends HelixError { override name = "HelixUnsupportedFeatureError"; }

export function wrapNativeError(err: any): HelixError {
  const status: number = err?.helixStatus ?? 0;
  const json: string = err?.helixErrorJson ?? "{}";
  let msg: string = err?.message ?? String(err);
  try { msg = JSON.parse(json)?.error?.message ?? msg; } catch {}

  const map: Record<number, new (...a: any[]) => HelixError> = {
    [-3]: HelixValidationError,
    [-4]: HelixModelNotFoundError,
    [-5]: HelixModelLoadFailedError,
    [-6]: HelixOomError,
    [-8]: HelixContextFullError,
    [-9]: HelixCancelledError,
    [-11]: HelixUnsupportedFeatureError,
  };
  const Cls = map[status] ?? HelixError;
  return new Cls(msg, status, json);
}
