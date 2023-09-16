#include "core/language_layer.h"

#include <SDL.h>
#include <SDL_events.h>
#include <SDL_video.h>
#include <core/mem.h>
#include <stdio.h>
#include <string.h>

typedef struct cst_window {
  SDL_Window*  window;
  SDL_Surface* surface;
  u32          width;
  u32          height;
} cst_window;

typedef struct v2i {
  i32 x, y;
} v2i;
typedef struct v2f {
  f32 x, y;
} v2f;

typedef struct v3i {
  i32 x, y, z;
} v3i;
typedef struct v3f {
  f32 x, y, z;
} v3f;

typedef struct v4f {
  f32 x, y, z, w;
} v4f;

typedef struct v4i {
  i32 x, y;
  union {
    struct {
      i32 z, w;
    };
    struct {
      i32 width, height;
    };
  };
} v4i;

typedef struct cst_rgba {
  u8 r, g, b, a;
} cst_rgba;

#define MAX_VERTICES       4096
#define MAX_INDICES        4096
#define MAX_VERTEX_ATTRIBS 8

typedef struct cst_vertex_shader_input {
  const byte* data[MAX_VERTEX_ATTRIBS];
} cst_vertex_shader_input;

typedef v4f cst_vertex_shader_output;

cst_vertex_shader_output
cst_interpolate_vertex(cst_vertex_shader_output v0, cst_vertex_shader_output v1, f32 t) {
  cst_vertex_shader_output result = {
      .x = v0.x * (1.0f - t) + v1.x * t,
      .y = v0.y * (1.0f - t) + v1.y * t,
      .z = v0.z * (1.0f - t) + v1.z * t,
      .w = v0.w * (1.0f - t) + v1.w * t,
  };
  return result;
}

typedef cst_vertex_shader_output (*cst_vertex_shader_func)(cst_vertex_shader_input in);

typedef struct cst_vertex_shader {
  i32                    n_vertex_attributes;
  cst_vertex_shader_func shader_func;
} cst_vertex_shader;

typedef struct cst_fragment_shader_input {
  v4f frag_pos;
} cst_fragment_shader_input;

typedef struct cst_fragment_shader_output {
  cst_rgba frag_color;
} cst_fragment_shader_output;

typedef cst_fragment_shader_output (*cst_fragment_shader_func)(cst_fragment_shader_input in);
typedef struct cst_fragment_shader {
  SDL_Surface*             surface;
  cst_fragment_shader_func shader_func;
  b32                      interpolate_z;
  b32                      interpolate_w;
} cst_fragment_shader;

void cst_put_pixel(SDL_Surface* surface, i32 x, i32 y, u32 value) {
  i32   bpp   = surface->format->BytesPerPixel;
  byte* pixel = (byte*)surface->pixels + y * surface->pitch + x * bpp;

  switch (bpp) {
  case 1:
    *pixel = value;
    break;
  case 2:
    *(u16*)pixel = value;
    break;
  case 3:
    if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
      pixel[0] = (value >> 16) & 0xff;
      pixel[1] = (value >> 8) & 0xff;
      pixel[2] = value & 0xff;
    } else {
      pixel[0] = value & 0xff;
      pixel[1] = (value >> 8) & 0xff;
      pixel[2] = (value >> 16) & 0xff;
    }

    break;
  case 4:
    *(u32*)pixel = value;
    break;
  };
}

typedef struct cst_triangle {
  union {
    struct {
      v2f a, b, c;
    };

    v2f v[3];
  };
} cst_triangle;

typedef struct cst_edge {
  f32 a, b, c;
  b32 tie;
} cst_edge;

cst_edge cst_edge_create(v4f v0, v4f v1) {
  cst_edge edge;
  edge.a   = v0.y - v1.y;
  edge.b   = v1.x - v0.x;
  edge.c   = -(edge.a * (v0.x + v1.x) + edge.b * (v0.y + v1.y)) * 0.5f;
  edge.tie = (edge.a != 0) ? edge.a > 0 : edge.b > 0;
  return edge;
};

f32 cst_edge_evaluate(cst_edge* edge, f32 x, f32 y) {
  return edge->a * x + edge->b * y + edge->c;
}

b32 cst_edge_test_value(cst_edge* edge, f32 v) {
  return (v > 0 || (v == 0 && edge->tie));
}

b32 cst_edge_test_point(cst_edge* edge, f32 x, f32 y) {
  return cst_edge_test_value(edge, cst_edge_evaluate(edge, x, y));
}

typedef enum cst_clip_mask {
  PosX = 0x01,
  NegX = 0x02,
  PosY = 0x04,
  NegY = 0x08,
  PosZ = 0x10,
  NegZ = 0x20,
} cst_clip_mask;

typedef enum cst_draw_mode { Point, Line, Triangle } cst_draw_mode;
typedef enum cst_cull_mode { None, CCW, CW } cst_cull_mode;

typedef struct cst_vertex_attribute {
  const byte* buffer;
  i32         stride;
} cst_vertex_attribute;

typedef struct cst_viewport {
  v4i bounds;
  f32 px, py, ox, oy;
} cst_viewport;

typedef struct cst_vertex_processor {
  cst_vertex_attribute vertex_attributes[MAX_VERTEX_ATTRIBS];
  u32                  n_vertex_attributes;
  cst_vertex_shader    vertex_shader;
  cst_viewport         viewport;
  f32                  near_plane;
  f32                  far_plane;
  cst_cull_mode        cull_mode;

  cst_vertex_shader_output out_vertices[MAX_VERTICES];
  i32                      out_indices[MAX_INDICES];
  i32                      vert_clip_masks[MAX_VERTICES];
  u32                      n_out_vertices;
  u32                      n_out_indices;
} cst_vertex_processor;

void cst_vertex_processor_set_viewport(
    cst_vertex_processor* v,
    i32                   x,
    i32                   y,
    i32                   width,
    i32                   height) {
  v->viewport.bounds.x      = x;
  v->viewport.bounds.y      = y;
  v->viewport.bounds.width  = width;
  v->viewport.bounds.height = height;

  v->viewport.px = width / 2.0f;
  v->viewport.py = height / 2.0f;
  v->viewport.ox = (x + v->viewport.px);
  v->viewport.oy = (y + v->viewport.py);
}

void cst_vertex_processor_set_depth_range(cst_vertex_processor* v, f32 near, f32 far) {
  v->near_plane = near;
  v->far_plane  = far;
}

void cst_vertex_processor_set_cull_mode(cst_vertex_processor* v, cst_cull_mode cull_mode) {
  v->cull_mode = cull_mode;
}

void cst_vertex_processor_set_vertex_shader(cst_vertex_processor* v, cst_vertex_shader shader) {
  v->vertex_shader = shader;
}

void cst_vertex_processor_set_vertex_attrib_ptr(
    cst_vertex_processor* v,
    u32                   idx,
    i32                   stride,
    byte*                 buffer) {
  Assert(idx < MAX_VERTEX_ATTRIBS);
  v->vertex_attributes[idx].buffer = buffer;
  v->vertex_attributes[idx].stride = stride;
}

const byte*
cst_vertex_processor_get_vertex_attrib_ptr(cst_vertex_processor* v, i32 attrib_idx, i32 elt_idx) {
  cst_vertex_attribute attrib = v->vertex_attributes[attrib_idx];

  u32 offset = attrib.stride * elt_idx;
  return &attrib.buffer[offset];
}

cst_vertex_shader_output
cst_vertex_processor_process_vertex(cst_vertex_processor* v, cst_vertex_shader_input in) {
  return v->vertex_shader.shader_func(in);
}

void cst_vertex_processor_transform_vertices(cst_vertex_processor* v) {
  // TODO(bryson): implement set of already processed vertices
  for (i32 i = 0; i < v->n_out_indices; ++i) {
    i32 idx = v->out_indices[i];

    if (idx == -1) continue;

    cst_vertex_shader_output* v_out = &v->out_vertices[idx];

    // perspective divide
    f32 inv_w = 1.0f / v_out->w;
    v_out->x *= inv_w;
    v_out->y *= inv_w;
    v_out->z *= inv_w;

    // viewport transform
    v_out->x = (v->viewport.px * v_out->x + v->viewport.ox);
    v_out->y = (v->viewport.py * -v_out->y + v->viewport.oy);
    v_out->z =
        0.5f * (v->far_plane - v->near_plane) * v_out->z + 0.5f * (v->near_plane + v->far_plane);
  }
}

i32 cst_vertex_clip_mask(cst_vertex_shader_output vert) {
  i32 mask = 0;

  if (vert.w - vert.x < 0) mask |= PosX;
  if (vert.x + vert.w < 0) mask |= NegX;
  if (vert.w - vert.y < 0) mask |= PosY;
  if (vert.y + vert.w < 0) mask |= NegY;
  if (vert.w - vert.z < 0) mask |= PosZ;
  if (vert.z + vert.w < 0) mask |= NegZ;

  return mask;
}

void cst_vertex_processor_clip_points(cst_vertex_processor* v) {
  for (i32 i = 0; i < v->n_out_vertices; ++i) {
    v->vert_clip_masks[i] = cst_vertex_clip_mask(v->out_vertices[i]);
  }

  for (i32 i = 0; i < v->n_out_indices; ++i) {
    if (v->vert_clip_masks[v->out_indices[i]]) v->out_indices[i] = -1;
  }
}

typedef struct cst_line_clipper {
  cst_vertex_shader_output v0, v1;
  f32                      t0, t1;
  b32                      clipped;
} cst_line_clipper;

cst_line_clipper cst_line_clipper_create(cst_vertex_shader_output v0, cst_vertex_shader_output v1) {
  cst_line_clipper line_clipper = {
      .v0      = v0,
      .v1      = v1,
      .t0      = 0.0f,
      .t1      = 1.0f,
      .clipped = false,
  };
  return line_clipper;
}

void cst_line_clipper_clip_to_plane(cst_line_clipper* clipper, f32 a, f32 b, f32 c, f32 d) {
  if (clipper->clipped) return;

  f32 dp0 = a * clipper->v0.x + b * clipper->v0.y + c * clipper->v0.z + d * clipper->v0.w;
  f32 dp1 = a * clipper->v1.x + b * clipper->v1.y + c * clipper->v1.z + d * clipper->v1.w;

  b32 dp0neg = dp0 < 0;
  b32 dp1neg = dp1 < 0;

  if (dp0neg && dp1neg) {
    clipper->clipped = true;
    return;
  }

  if (dp0neg) {
    f32 t       = -dp0 / (dp1 - dp0);
    clipper->t0 = Max(clipper->t0, t);
  } else {
    f32 t       = dp0 / (dp0 - dp1);
    clipper->t1 = Min(clipper->t1, t);
  }
}

typedef struct cst_triangle_clipper {
  i32                       in_indices[MAX_INDICES];
  u32                       n_in_indices;
  i32                       out_indices[MAX_INDICES];
  u32                       n_out_indices;
  b32                       clipped;
  cst_vertex_shader_output* vertices;
  u32*                      n_vertices;
} cst_triangle_clipper;

cst_triangle_clipper cst_triangle_clipper_create(
    cst_vertex_shader_output* vertices,
    u32*                      n_vertices,
    i32                       idx0,
    i32                       idx1,
    i32                       idx2) {
  cst_triangle_clipper clipper = {
      .in_indices    = {idx0, idx1, idx2},
      .n_in_indices  = 3,
      .out_indices   = {},
      .n_out_indices = 0,
      .vertices      = vertices,
      .n_vertices    = n_vertices,
  };
  return clipper;
}

void cst_triangle_clipper_clip_to_plane(cst_triangle_clipper* clipper, f32 a, f32 b, f32 c, f32 d) {
  if (clipper->n_in_indices < 3) {
    clipper->clipped = true;
    return;
  }

  clipper->n_out_indices = 0;

  i32 prev_idx                                 = clipper->in_indices[0];
  clipper->in_indices[clipper->n_in_indices++] = prev_idx;

  cst_vertex_shader_output prev_vert = clipper->vertices[prev_idx];
  f32 prev_dp = a * prev_vert.x + b * prev_vert.y + c * prev_vert.z + d * prev_vert.w;

  for (i32 i = 1; i < clipper->n_in_indices; ++i) {
    i32                      idx = clipper->in_indices[i];
    cst_vertex_shader_output v   = clipper->vertices[idx];
    f32                      dp  = a * v.x + b * v.y + c * v.z + d * v.w;

    if (prev_dp >= 0) clipper->out_indices[clipper->n_out_indices++] = prev_idx;

    if (Sign(dp) != Sign(prev_dp)) {
      f32 t = (dp < 0) ? prev_dp / (prev_dp - dp) : -prev_dp / (dp - prev_dp);

      cst_vertex_shader_output interp_v =
          cst_interpolate_vertex(clipper->vertices[prev_idx], clipper->vertices[idx], t);
      clipper->vertices[(*clipper->n_vertices)++]    = interp_v;
      clipper->out_indices[clipper->n_out_indices++] = *clipper->n_vertices - 1;
    }

    prev_idx = idx;
    prev_dp  = dp;
  }

  i32 tmp[3] = {clipper->in_indices[0], clipper->in_indices[1], clipper->in_indices[2]};

  clipper->in_indices[0] = clipper->out_indices[0];
  clipper->in_indices[1] = clipper->out_indices[1];
  clipper->in_indices[2] = clipper->out_indices[2];

  clipper->out_indices[0] = tmp[0];
  clipper->out_indices[1] = tmp[1];
  clipper->out_indices[2] = tmp[2];
}

void cst_vertex_processor_clip_lines(cst_vertex_processor* v) {
  for (i32 i = 0; i < v->n_out_vertices; ++i) {
    v->vert_clip_masks[i] = cst_vertex_clip_mask(v->out_vertices[i]);
  }

  for (i32 i = 0; i < v->n_out_indices; i += 2) {
    i32 idx0 = v->out_indices[i];
    i32 idx1 = v->out_indices[i + 1];

    cst_vertex_shader_output v0 = v->out_vertices[idx0];
    cst_vertex_shader_output v1 = v->out_vertices[idx1];

    i32 clip_mask = v->vert_clip_masks[idx0] | v->vert_clip_masks[idx1];

    cst_line_clipper clipper = cst_line_clipper_create(v0, v1);
    if (clip_mask & PosX) cst_line_clipper_clip_to_plane(&clipper, -1, 0, 0, 1);
    if (clip_mask & NegX) cst_line_clipper_clip_to_plane(&clipper, 1, 0, 0, 1);
    if (clip_mask & PosY) cst_line_clipper_clip_to_plane(&clipper, 0, -1, 0, 1);
    if (clip_mask & NegY) cst_line_clipper_clip_to_plane(&clipper, 0, 1, 0, 1);
    if (clip_mask & PosZ) cst_line_clipper_clip_to_plane(&clipper, 0, 0, -1, 1);
    if (clip_mask & NegZ) cst_line_clipper_clip_to_plane(&clipper, 0, 0, 1, 1);

    if (clipper.clipped) {
      v->out_indices[i]     = -1;
      v->out_indices[i + 1] = -1;
      continue;
    }

    if (v->vert_clip_masks[idx0]) {
      cst_vertex_shader_output interp_v    = cst_interpolate_vertex(v0, v1, clipper.t0);
      v->out_vertices[v->n_out_vertices++] = interp_v;
      v->out_indices[i]                    = v->n_out_vertices - 1;
    }

    if (v->vert_clip_masks[idx1]) {
      cst_vertex_shader_output interp_v    = cst_interpolate_vertex(v0, v1, clipper.t1);
      v->out_vertices[v->n_out_vertices++] = interp_v;
      v->out_indices[i + 1]                = v->n_out_vertices - 1;
    }
  }
}

void cst_vertex_processor_clip_triangles(cst_vertex_processor* v) {
  for (i32 i = 0; i < v->n_out_vertices; ++i) {
    v->vert_clip_masks[i] = cst_vertex_clip_mask(v->out_vertices[i]);
  }

  u32 N = v->n_out_indices;
  for (i32 i = 0; i < N; i += 3) {
    i32 idx0 = v->out_indices[i];
    i32 idx1 = v->out_indices[i + 1];
    i32 idx2 = v->out_indices[i + 2];

    i32 clip_mask = v->vert_clip_masks[idx0] | v->vert_clip_masks[idx1] | v->vert_clip_masks[idx2];

    cst_triangle_clipper clipper =
        cst_triangle_clipper_create(v->out_vertices, &v->n_out_vertices, idx0, idx1, idx2);
    if (clip_mask & PosX) cst_triangle_clipper_clip_to_plane(&clipper, -1, 0, 0, 1);
    if (clip_mask & NegX) cst_triangle_clipper_clip_to_plane(&clipper, 1, 0, 0, 1);
    if (clip_mask & PosY) cst_triangle_clipper_clip_to_plane(&clipper, 0, -1, 0, 1);
    if (clip_mask & NegY) cst_triangle_clipper_clip_to_plane(&clipper, 0, 1, 0, 1);
    if (clip_mask & PosZ) cst_triangle_clipper_clip_to_plane(&clipper, 0, 0, -1, 1);
    if (clip_mask & NegZ) cst_triangle_clipper_clip_to_plane(&clipper, 0, 0, 1, 1);

    if (clipper.clipped) {
      v->out_indices[i]     = -1;
      v->out_indices[i + 1] = -1;
      v->out_indices[i + 2] = -1;
      continue;
    }

    v->out_indices[i]     = clipper.in_indices[0];
    v->out_indices[i + 1] = clipper.in_indices[1];
    v->out_indices[i + 2] = clipper.in_indices[2];

    for (i32 idx = 3; idx < clipper.n_in_indices; ++idx) {
      v->out_indices[v->n_out_indices++] = clipper.in_indices[0];
      v->out_indices[v->n_out_indices++] = clipper.in_indices[idx - 1];
      v->out_indices[v->n_out_indices++] = clipper.in_indices[idx];
    }
  }
}

void cst_vertex_processor_cull_triangles(cst_vertex_processor* v) {
  for (i32 i = 0; i <= v->n_out_indices; i += 3) {
    if (v->out_indices[i] == -1) continue;

    cst_vertex_shader_output* v0 = &v->out_vertices[v->out_indices[i]];
    cst_vertex_shader_output* v1 = &v->out_vertices[v->out_indices[i + 1]];
    cst_vertex_shader_output* v2 = &v->out_vertices[v->out_indices[i + 2]];

    f32 facing = (v0->x - v1->x) * (v2->y - v1->y) - (v2->x - v1->x) * (v0->y - v1->y);

    if (facing < 0) {
      if (v->cull_mode == CW) {
        v->out_indices[i] = v->out_indices[i + 1] = v->out_indices[i + 2] = -1;
      }
    } else {
      if (v->cull_mode == CCW) {
        v->out_indices[i] = v->out_indices[i + 1] = v->out_indices[i + 2] = -1;
      } else {
        i32 tmp               = v->out_indices[i];
        v->out_indices[i]     = v->out_indices[i + 2];
        v->out_indices[i + 2] = tmp;
      }
    }
  }
}

void cst_vertex_processor_clip_primitives(cst_vertex_processor* v, cst_draw_mode mode) {
  switch (mode) {
  case Triangle:
    cst_vertex_processor_clip_triangles(v);
    break;
  case Line:
    cst_vertex_processor_clip_lines(v);
    break;
  case Point:
    cst_vertex_processor_clip_points(v);
    break;
  }
}

typedef struct cst_rasterizer {
  v4i scissor_rect;
} cst_rasterizer;

void cst_rasterizer_set_scissor_rect(cst_rasterizer* r, i32 x, i32 y, i32 width, i32 height) {
  r->scissor_rect.x      = x;
  r->scissor_rect.y      = y;
  r->scissor_rect.width  = width;
  r->scissor_rect.height = height;
}

b32 cst_rasterizer_scissor_test(cst_rasterizer* r, f32 x, f32 y) {
  return x >= r->scissor_rect.x && x < r->scissor_rect.x + r->scissor_rect.width &&
         y >= r->scissor_rect.y && y < r->scissor_rect.y + r->scissor_rect.height;
}

cst_fragment_shader_input cst_fragment_shader_data_from_vertex(cst_fragment_shader* frag, v4f v) {
  cst_fragment_shader_input input = {
      .frag_pos =
          {
              .x = (i32)v.x,
              .y = (i32)v.y,
              .z = (frag->interpolate_z) ? v.z : 0,
              .w = (frag->interpolate_w) ? v.w : 0,
          },
  };
  return input;
}

typedef struct cst_fragment_processor {
  cst_rasterizer      rasterizer;
  cst_fragment_shader fragment_shader;
} cst_fragment_processor;

cst_fragment_shader_output
cst_fragment_processor_process_fragment(cst_fragment_processor* f, cst_fragment_shader_input in) {
  return f->fragment_shader.shader_func(in);
}

void cst_fragment_processor_put_pixel(cst_fragment_processor* f, cst_fragment_shader_input in) {
  cst_fragment_shader_output frag_out   = cst_fragment_processor_process_fragment(f, in);
  cst_rgba                   frag_color = frag_out.frag_color;

  u32 color =
      SDL_MapRGB(f->fragment_shader.surface->format, frag_color.r, frag_color.g, frag_color.b);
  cst_put_pixel(f->fragment_shader.surface, in.frag_pos.x, in.frag_pos.y, color);
}

void cst_fragment_processor_set_fragment_shader(
    cst_fragment_processor* f,
    cst_fragment_shader     shader) {
  f->fragment_shader = shader;
}
void cst_fragment_processor_set_rasterizer(cst_fragment_processor* f, cst_rasterizer rasterizer) {
  f->rasterizer = rasterizer;
}

void cst_fragment_processor_draw_point(cst_fragment_processor* f, v4f point) {
  if (!cst_rasterizer_scissor_test(&f->rasterizer, point.x, point.y)) return;

  cst_fragment_shader_input frag_in =
      cst_fragment_shader_data_from_vertex(&f->fragment_shader, point);
  cst_fragment_processor_put_pixel(f, frag_in);
}

void cst_fragment_processor_draw_point_list(
    cst_fragment_processor* f,
    v4f*                    vertices,
    const i32*              indices,
    u32                     n_indices) {
  for (i32 i = 0; i < n_indices; ++i) {
    if (indices[i] == -1) continue;
    cst_fragment_processor_draw_point(f, vertices[indices[i]]);
  }
}

void cst_fragment_processor_draw_line(cst_fragment_processor* f, v4f v0, v4f v1) {
  i32 adx   = abs((i32)v1.x - (i32)v0.x);
  i32 ady   = abs((i32)v1.y - (i32)v0.y);
  i32 steps = Max(adx, ady);

  f32 inv_steps = 1.0f / steps;
  v4f step      = {
           .x = (v1.x - v0.x) * inv_steps,
           .y = (v1.y - v0.y) * inv_steps,
           .z = (f->fragment_shader.interpolate_z) ? (v1.z - v0.z) * inv_steps : 0,
           .w = (f->fragment_shader.interpolate_w) ? (v1.w - v0.w) * inv_steps : 0,
  };

  v4f v = v0;
  while (steps-- > 0) {
    cst_fragment_shader_input frag_in =
        cst_fragment_shader_data_from_vertex(&f->fragment_shader, v);

    if (cst_rasterizer_scissor_test(&f->rasterizer, v.x, v.y))
      cst_fragment_processor_put_pixel(f, frag_in);

    v.x += step.x;
    v.y += step.y;
    if (f->fragment_shader.interpolate_z) v.z += step.z;
    if (f->fragment_shader.interpolate_w) v.w += step.w;
  }
}

void cst_fragment_processor_draw_line_list(
    cst_fragment_processor* f,
    v4f*                    vertices,
    const i32*              indices,
    u32                     n_indices) {
  for (i32 i = 0; i <= n_indices - 2; i += 2) {
    if (indices[i] == -1) continue;
    cst_fragment_processor_draw_line(f, vertices[indices[i]], vertices[indices[i + 1]]);
  }
}

void cst_fragment_processor_draw_triangle(cst_fragment_processor* f, v4f v0, v4f v1, v4f v2) {
  i32 min_x = Min(Min(v0.x, v1.x), v2.x);
  i32 max_x = Max(Max(v0.x, v1.x), v2.x);
  i32 min_y = Min(Min(v0.y, v1.y), v2.y);
  i32 max_y = Max(Max(v0.y, v1.y), v2.y);

  min_x = Max(min_x, f->rasterizer.scissor_rect.x);
  max_x = Min(max_x, f->rasterizer.scissor_rect.x + f->rasterizer.scissor_rect.width);
  min_y = Max(min_y, f->rasterizer.scissor_rect.y);
  max_y = Min(max_y, f->rasterizer.scissor_rect.y + f->rasterizer.scissor_rect.height);

  cst_edge e0 = cst_edge_create(v1, v2);
  cst_edge e1 = cst_edge_create(v2, v0);
  cst_edge e2 = cst_edge_create(v0, v1);

  f32 area = 0.5 * (e0.c + e1.c + e2.c);

  if (area < 0) return;  // back face culling

  for (f32 x = min_x + 0.5f, xm = max_x + 0.5f; x <= xm; x += 1.0f) {
    for (f32 y = min_y + 0.5f, ym = max_y + 0.5f; y <= ym; y += 1.0f) {
      if (cst_edge_test_point(&e0, x, y) && cst_edge_test_point(&e1, x, y) &&
          cst_edge_test_point(&e2, x, y)) {
        v4f                       v = {.x = x, .y = y};
        cst_fragment_shader_input frag_in =
            cst_fragment_shader_data_from_vertex(&f->fragment_shader, v);
        cst_fragment_processor_put_pixel(f, frag_in);
      }
    }
  }
}

void cst_fragment_processor_draw_triangle_list(
    cst_fragment_processor* f,
    v4f*                    vertices,
    const i32*              indices,
    u32                     n_indices) {
  for (i32 i = 0; i <= n_indices - 3; i += 3) {
    if (indices[i] == -1) continue;
    cst_fragment_processor_draw_triangle(
        f,
        vertices[indices[i]],
        vertices[indices[i + 1]],
        vertices[i + 2]);
  }
}

typedef struct cst_renderer {
  cst_vertex_processor   vertex_processor;
  cst_fragment_processor fragment_processor;
} cst_renderer;

void cst_renderer_draw_primitives(cst_renderer* renderer, cst_draw_mode mode) {
  switch (mode) {
  case Triangle:
    cst_vertex_processor_cull_triangles(&renderer->vertex_processor);
    cst_fragment_processor_draw_triangle_list(
        &renderer->fragment_processor,
        renderer->vertex_processor.out_vertices,
        renderer->vertex_processor.out_indices,
        renderer->vertex_processor.n_out_indices);
    break;
  case Line:
    cst_fragment_processor_draw_line_list(
        &renderer->fragment_processor,
        renderer->vertex_processor.out_vertices,
        renderer->vertex_processor.out_indices,
        renderer->vertex_processor.n_out_indices);
    break;
  case Point:
    cst_fragment_processor_draw_point_list(
        &renderer->fragment_processor,
        renderer->vertex_processor.out_vertices,
        renderer->vertex_processor.out_indices,
        renderer->vertex_processor.n_out_indices);
    break;
  }
}

void cst_renderer_clear(cst_renderer* renderer, i32 r, i32 g, i32 b) {
  SDL_Surface* surface = renderer->fragment_processor.fragment_shader.surface;
  u32          color   = SDL_MapRGB(surface->format, r, g, b);
  i32          bpp     = surface->format->BytesPerPixel;
  memset(surface->pixels, color, surface->w * surface->h * bpp);
}

void cst_renderer_process_primitives(cst_renderer* renderer, cst_draw_mode mode) {
  cst_vertex_processor_clip_primitives(&renderer->vertex_processor, mode);
  cst_vertex_processor_transform_vertices(&renderer->vertex_processor);
  cst_renderer_draw_primitives(renderer, mode);
}

cst_vertex_shader_input
cst_vertex_processor_prepare_shader_input(cst_vertex_processor* v, i32 idx) {
  cst_vertex_shader_input in = {};
  for (i32 i = 0; i < v->vertex_shader.n_vertex_attributes; ++i) {
    in.data[i] = cst_vertex_processor_get_vertex_attrib_ptr(v, i, idx);
  }
  return in;
}

void cst_renderer_draw_elements(
    cst_renderer* renderer,
    cst_draw_mode mode,
    u32           count,
    i32*          indices) {
  cst_vertex_processor* v = &renderer->vertex_processor;
  v->n_out_vertices       = 0;
  v->n_out_indices        = 0;

  for (i32 i = 0; i < count; ++i) {
    i32 index = indices[i];

    cst_vertex_shader_input v_in =
        cst_vertex_processor_prepare_shader_input(&renderer->vertex_processor, index);

    i32 out_idx                        = v->n_out_vertices;
    v->out_indices[v->n_out_indices++] = out_idx;

    v->out_vertices[v->n_out_vertices++] = cst_vertex_processor_process_vertex(v, v_in);
  }

  cst_renderer_process_primitives(renderer, mode);
}

cst_window cst_window_create(str name, u32 width, u32 height) {
  SDL_Window* sdl_window = SDL_CreateWindow(
      name,
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      width,
      height,
      SDL_WINDOW_SHOWN);
  SDL_Surface* surface = SDL_GetWindowSurface(sdl_window);

  cst_window window = {
      .window  = sdl_window,
      .surface = surface,
      .width   = width,
      .height  = height,
  };
  return window;
}

typedef struct cst_app {
  cst_window   window;
  b32          running;
  Arena        program_memory;
  cst_renderer renderer;
} cst_app;

cst_vertex_shader_output default_vert(cst_vertex_shader_input in) {
  cst_vertex_shader_output out = {};

  static f32 t    = 0;
  const v4f* data = (v4f*)in.data[0];
  out.x           = data->x + cos(t / 1'000.0f) * 0.25f;
  out.y           = data->y + sin(t / 1'000.0f) * 0.25f;
  out.z           = data->z;
  out.w           = 1.0f;

  t++;

  return out;
};

cst_fragment_shader_output default_frag(cst_fragment_shader_input in) {
  cst_fragment_shader_output out = {};
  out.frag_color.r               = 255;
  out.frag_color.g               = 255;
  out.frag_color.b               = 0;
  out.frag_color.a               = 255;
  return out;
}

cst_app cst_app_create(str name, u32 width, u32 height) {
  SDL_Init(SDL_INIT_VIDEO);
  cst_window window = cst_window_create(name, width, height);

  cst_vertex_shader vertex_shader = {
      .shader_func         = default_vert,
      .n_vertex_attributes = 1,
  };

  cst_vertex_processor V;
  cst_vertex_processor_set_viewport(&V, 0, 0, width, height);
  cst_vertex_processor_set_cull_mode(&V, CCW);
  cst_vertex_processor_set_depth_range(&V, 0.1f, 100.0f);
  cst_vertex_processor_set_vertex_shader(&V, vertex_shader);

  cst_rasterizer raster;
  cst_rasterizer_set_scissor_rect(&raster, 0, 0, width, height);

  cst_fragment_shader fragment_shader = {
      .shader_func   = default_frag,
      .surface       = window.surface,
      .interpolate_z = false,
      .interpolate_w = false,
  };

  cst_fragment_processor F;
  cst_fragment_processor_set_fragment_shader(&F, fragment_shader);
  cst_fragment_processor_set_rasterizer(&F, raster);

  cst_renderer renderer = {
      .vertex_processor   = V,
      .fragment_processor = F,
  };

  cst_app app = {
      .window         = window,
      .renderer       = renderer,
      .running        = true,
      .program_memory = arena_create(Kilobytes(4)),
  };

  return app;
}

void cst_app_run(cst_app* app) {
  while (app->running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) { app->running = false; }
    }

    v4f p0 = {.x = -0.5f, .y = -0.5f, .z = 0, .w = 0};
    v4f p1 = {.x = -.5f, .y = 0.5f, .z = 0, .w = 0};
    v4f p2 = {.x = 0.5f, .y = 0.5f, .z = 0, .w = 0};

    v4f vertices[3] = {p0, p1, p2};
    i32 indices[3]  = {0, 1, 2};

    cst_renderer_clear(&app->renderer, 0, 0, 0);

    cst_vertex_processor_set_vertex_attrib_ptr(
        &app->renderer.vertex_processor,
        0,               // index
        sizeof(v4f),     // stride
        (byte*)vertices  // buffer
    );
    cst_renderer_draw_elements(&app->renderer, Triangle, 3, indices);

    SDL_UpdateWindowSurface(app->window.window);
  }
}

void cst_app_destroy(cst_app* app) {
  arena_release(&app->program_memory);
  SDL_DestroyWindow(app->window.window);
  SDL_Quit();
}

int main(void) {
  cst_app app = cst_app_create("Constellation", 640, 480);
  cst_app_run(&app);
  cst_app_destroy(&app);
}
