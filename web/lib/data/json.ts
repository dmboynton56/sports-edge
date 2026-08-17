export type JsonPrimitive = null | boolean | number | string;
export type JsonValue = JsonPrimitive | JsonArray | JsonObject;
export type JsonArray = JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

const objectTag = <Value>(value: Value) => Object.prototype.toString.call(value);

export function isFiniteNumber<Value>(value: Value): value is Value & number {
  return Number.isFinite(value);
}

export function isJsonObject<Value>(value: Value): value is Value & JsonObject {
  return value !== null && objectTag(value) === "[object Object]";
}

export function isJsonString<Value>(value: Value): value is Value & string {
  return objectTag(Object(value)) === "[object String]";
}

export function isJsonNumber<Value>(value: Value): value is Value & number {
  return objectTag(Object(value)) === "[object Number]" && isFiniteNumber(value);
}

export function isJsonBoolean<Value>(value: Value): value is Value & boolean {
  return objectTag(Object(value)) === "[object Boolean]";
}

export function isJsonValue<Value>(value: Value): value is Value & JsonValue {
  if (value === null || isJsonString(value) || isJsonNumber(value) || isJsonBoolean(value)) {
    return true;
  }
  if (Array.isArray(value)) return value.every(isJsonValue);
  return isJsonObject(value) && Object.values(value).every(isJsonValue);
}

export function parseJson(text: string): JsonValue {
  const value = JSON.parse(text);
  if (!isJsonValue(value)) throw new Error("Expected a JSON-compatible value.");
  return value;
}
