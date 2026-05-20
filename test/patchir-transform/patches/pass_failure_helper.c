// RUN: true
// Helper patch C body shared by pass_failure_tests.json.  Compiles
// cleanly and defines exactly one symbol (`patch__before__present_symbol`),
// so any spec naming a different `function_name` exercises the
// missing-symbol branch in PatchOperationImpl::ensurePatchFunctionAvailable.

void patch__before__present_symbol(unsigned int a, unsigned short b)
{
    (void) a;
    (void) b;
}
