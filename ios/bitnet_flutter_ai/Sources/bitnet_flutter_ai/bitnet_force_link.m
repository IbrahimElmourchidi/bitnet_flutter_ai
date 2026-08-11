// Keep-alive references for the bitnet_bridge static library.
//
// The bn_* entry points are reached ONLY through Dart FFI
// (DynamicLibrary.process() → dlsym) — no Swift/ObjC code calls them at link
// time. Under use_frameworks! the plugin is a dynamic framework built from its
// own sources, and the linker dead-strips the vendored libbitnet_bridge.a
// unless something in the framework references its symbols. dlsym then fails
// with "symbol not found: bn_init".
//
// Referencing each exported function here forces the linker to pull the
// archive's objects into the framework. This is more robust than a
// `-force_load` linker flag pointed at the xcframework's staged .a path, which
// Xcode treats as a build-input dependency and cannot resolve on a clean build
// (the [CP] Copy XCFrameworks phase stages the .a after the link step tries to
// validate it). This shim needs no path and no build-phase ordering.
//
// Declared extern here (rather than #import "bitnet_bridge.h") so this shim
// does not depend on the vendored xcframework's Headers being on the ObjC
// include path. Only the addresses are taken; the signatures need not match
// the real ones for the linker to retain the symbols, but we keep them
// accurate for clarity.

#include <stdint.h>
#include <stddef.h>

extern void *bn_init(const char *, int32_t, int32_t, char **);
extern void bn_free(void *);
extern void bn_free_string(char *);
extern int32_t bn_prompt(void *, const char *, int32_t, char **);
extern char *bn_next_token(void *, int32_t *, char **);
extern int32_t bn_context_size(const void *);
extern const char *bn_meta(const void *, const char *);

__attribute__((used))
const void *bitnet_force_link_refs[] = {
    (const void *)&bn_init,
    (const void *)&bn_free,
    (const void *)&bn_free_string,
    (const void *)&bn_prompt,
    (const void *)&bn_next_token,
    (const void *)&bn_context_size,
    (const void *)&bn_meta,
};
