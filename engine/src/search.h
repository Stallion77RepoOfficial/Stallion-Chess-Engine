#pragma once
#include "movepick.h"
#include "nnue.h"
#include "params.h"
#include "position.h"
// tm.h removed; adjust_soft_limit forward declared in utils.h
#include "utils.h"
#include "../fathom/src/tbprobe.h"

// Global state for tablebase
extern bool tb_initialized;

// Forward declarations
inline int64_t safe_elapsed(const std::chrono::steady_clock::time_point &start){
  auto ms = time_elapsed(start);
  return ms ? ms : 1;
}

int analyze_sacrifice(Position &position, ThreadInfo &thread_info, int depth, int ply, int sacrificer_color);

// Probe WDL tablebase during search for positions with ≤6 pieces.
// Honors 50-move rule if thread_info.syzygy_50_move_rule is true, otherwise ignores halfmove clock for probing.
// Returns ScoreNone on probe failure, otherwise a normalized internal score (including 0 for draw).
int probe_wdl_tb(Position &position, const ThreadInfo &thread_info) {
  // Lightweight mid-node Syzygy WDL probe similar in spirit to Stockfish.
  // Differences from earlier version:
  //  * We always attempt a probe (within limits) and pass the rule50 clock instead of aborting early.
  //  * We respect both the compiled TB_LARGEST and user syzygy_probe_limit.
  //  * We map WDL (including CURSED/BLESSED variants) to stable near-mate scores.
  if (!tb_initialized || !thread_info.use_syzygy) return ScoreNone;
  // Remaining depth gating (simple heuristic): avoid probing when too close to leaf
  if (thread_info.syzygy_probe_depth > 0 && thread_info.max_iter_depth > 0) {
    int remaining = thread_info.max_iter_depth - thread_info.search_ply;
    if (remaining < thread_info.syzygy_probe_depth) return ScoreNone;
  }

  int piece_count = pop_count(position.colors_bb[0] | position.colors_bb[1]);
  int compiled_limit = TB_LARGEST ? (int)TB_LARGEST : 7;
  if (piece_count > compiled_limit) return ScoreNone;
  if (piece_count > thread_info.syzygy_probe_limit) return ScoreNone; // user limit

  // Tablebases are only defined for positions without castling rights.
  unsigned castling = 0;
  if (position.castling_squares[Colors::White][Sides::Kingside] != SquareNone) castling |= TB_CASTLING_K;
  if (position.castling_squares[Colors::White][Sides::Queenside] != SquareNone) castling |= TB_CASTLING_Q;
  if (position.castling_squares[Colors::Black][Sides::Kingside] != SquareNone) castling |= TB_CASTLING_k;
  if (position.castling_squares[Colors::Black][Sides::Queenside] != SquareNone) castling |= TB_CASTLING_q;
  if (castling) return ScoreNone;

  unsigned ep = position.ep_square != SquareNone ? position.ep_square : 0;
  // rule50 parameter: halfmove clock if honoring 50-move rule, else 0 to ignore it
  unsigned rule50 = thread_info.syzygy_50_move_rule ? position.halfmoves : 0;

  unsigned result = tb_probe_wdl(
      position.colors_bb[0], position.colors_bb[1],
      position.pieces_bb[PieceTypes::King], position.pieces_bb[PieceTypes::Queen],
      position.pieces_bb[PieceTypes::Rook], position.pieces_bb[PieceTypes::Bishop],
      position.pieces_bb[PieceTypes::Knight], position.pieces_bb[PieceTypes::Pawn],
      rule50, castling, ep, position.color);
  if (result == TB_RESULT_FAILED) { thread_data.tb_fails++; return ScoreNone; }
  thread_data.tb_hits++;
  int wdl = TB_GET_WDL(result);
  switch (wdl) {
    case TB_WIN:          return  TB_WIN_SCORE;
    case TB_CURSED_WIN:   return  TB_CURSED_WIN_SCORE;
    case TB_DRAW:         return  TB_DRAW_SCORE;
    case TB_BLESSED_LOSS: return  TB_BLESSED_LOSS_SCORE;
    case TB_LOSS:         return  TB_LOSS_SCORE;
    default:              return ScoreNone;
  }
}

constexpr int NormalizationFactor = 195;

void update_history(int16_t &entry, int score) { // Update history score
  entry += score - entry * abs(score) / 16384;
}
void update_corrhist(int16_t &entry, int score) { // Update history score
  entry += score - entry * abs(score) / 1024;
}

bool out_of_time(ThreadInfo &thread_info) {
  if (thread_data.stop || thread_info.datagen_stop) return true;
  // Only main thread performs global time/node budget checks.
  if (thread_info.thread_id != 0) return false;

  // Aggregate node count across all threads for node limit decisions.
  uint64_t total_nodes = thread_info.nodes.load();
  for (auto &ti : thread_data.thread_infos) total_nodes += ti.nodes.load();
  if (total_nodes >= thread_info.max_nodes_searched) {
    if (thread_info.doing_datagen) thread_info.datagen_stop = true; else thread_data.stop = true;
    return true;
  }

  // More frequent (adaptive) time checks.
  thread_info.time_checks++;
  const uint16_t check_interval = 512; // smaller than 1024 for finer granularity
  if (thread_info.time_checks >= check_interval) {
    thread_info.time_checks = 0;
    if (!thread_info.infinite_search && !thread_info.pondering) {
      uint64_t elapsed = time_elapsed(thread_info.start_time);
      thread_info.time_manager.update_node_count(total_nodes);
      bool in_trouble = false; // Placeholder for future eval trend detection
      if (thread_info.time_manager.should_stop(elapsed, thread_info.best_move_stable, in_trouble) ||
          elapsed > thread_info.max_time) {
        thread_data.stop = true;
        return true;
      }
    }
  }
  return false;
}

int16_t material_eval(const Position &position) {
  int m = (position.material_count[0] - position.material_count[1]) * 100 +
          (position.material_count[2] - position.material_count[3]) * 300 +
          (position.material_count[4] - position.material_count[5]) * 300 +
          (position.material_count[6] - position.material_count[7]) * 500 +
          (position.material_count[8] - position.material_count[9]) * 900;

  return position.color ? -m : m;
}

bool has_non_pawn_material(const Position &position, int color) {
  int s_indx = 2 + color;
  return (position.material_count[s_indx] ||
          position.material_count[s_indx + 2] ||
          position.material_count[s_indx + 4] ||
          position.material_count[s_indx + 6]);
}

int16_t total_mat_color(const Position &position, int color) {
  // total material for one color

  int m = 0;
  for (int i = 0; i < 5; i++) {
    m += position.material_count[i * 2 + color] * SeeValues[i + 1];
  }
  return m;
}

int eval(Position &position, ThreadInfo &thread_info) {
  int color = position.color;
  
  // First check if we can use tablebase WDL score
  if (thread_info.use_syzygy && tb_initialized) {
  int tb_score = probe_wdl_tb(position, thread_info);
    if (tb_score != ScoreNone) return tb_score;
  }
  
  int nnue_eval = thread_info.nnue_state.evaluate(color, thread_info.phase);

  int bonus2 = 0, bonus3 = 0, bonus4 = 0, bonus5 = 0;
  bool our_side = (thread_info.search_ply % 2 == 0);

  int start_index = std::max(thread_info.game_ply - thread_info.search_ply, 0);
  int s_m = thread_info.game_hist[start_index].m_diff;
  int sacrifice_pattern = 0;

  // Detect sacrifice patterns in recent history (simple heuristic patterns)
  for (int idx = start_index + 2; idx < thread_info.game_ply - 4; idx += 2) {
    bool pattern = (thread_info.game_hist[idx].m_diff < s_m &&
                    thread_info.game_hist[idx + 1].m_diff > s_m &&
                    thread_info.game_hist[idx + 2].m_diff < s_m &&
                    thread_info.game_hist[idx + 3].m_diff > s_m &&
                    thread_info.game_hist[idx + 4].m_diff < s_m);
    if (pattern) {
      sacrifice_pattern = s_m + thread_info.game_hist[idx + 4].m_diff;
      break;
    }
    // Big piece sacrifice (queen/rook capture) immediately after loss
    if ((thread_info.game_hist[idx].piece_moved == Pieces::WQueen ||
         thread_info.game_hist[idx].piece_moved == Pieces::BQueen ||
         thread_info.game_hist[idx].piece_moved == Pieces::WRook ||
         thread_info.game_hist[idx].piece_moved == Pieces::BRook) &&
        thread_info.game_hist[idx].is_cap) {
      sacrifice_pattern = 3;
      break;
    }
    // Multiple sacrifices in a short span
    if (idx < thread_info.game_ply - 6) {
      int sacrifice_count = 0;
      for (int i = idx; i < idx + 6 && i < thread_info.game_ply; i++) {
        if (i > 0 && thread_info.game_hist[i].m_diff <
                         thread_info.game_hist[i - 1].m_diff - 50) {
          sacrifice_count++;
        }
      }
      if (sacrifice_count >= 2) {
        sacrifice_pattern = 4;
        break;
      }
    }
  }

  int total_material = total_mat(position);

  if (sacrifice_pattern && total_material > 3000) {
    if (thread_info.search_ply % 2) {
      bonus2 = -55 * (nnue_eval < -250 ? 3 : nnue_eval < 0 ? 2 : 1);
    } else {
      bonus2 = 55 * (nnue_eval > 250 ? 3 : nnue_eval > 0 ? 2 : 1);
    }
    int king_pos = get_king_pos(position, color ^ 1);
    int king_file = get_file(king_pos);
    if ((king_file >= 4 && sacrifice_pattern == 3) ||
        (king_file < 4 && sacrifice_pattern == 3)) {
      bonus2 += (thread_info.search_ply % 2) ? -25 : 25;
    }
    if (sacrifice_pattern == 4) {
      bonus2 += (thread_info.search_ply % 2) ? -40 : 40;
    }
  }

  // Central control
  uint64_t center_squares = (1ULL << 27) | (1ULL << 28) | (1ULL << 35) | (1ULL << 36);
  uint64_t extended_center = center_squares | (1ULL << 26) | (1ULL << 29) | (1ULL << 34) |
                             (1ULL << 37) | (1ULL << 42) | (1ULL << 43) | (1ULL << 44) | (1ULL << 45);
  int center_control = 0;
  if (position.pieces_bb[PieceTypes::Knight] & position.colors_bb[color] & extended_center)
    center_control += 15;
  if (position.pieces_bb[PieceTypes::Bishop] & position.colors_bb[color] & extended_center)
    center_control += 12;
  if (position.pieces_bb[PieceTypes::Pawn] & position.colors_bb[color] & center_squares)
    center_control += 10;

  int mobility_bonus = 0;
  if (thread_info.game_ply < 20) {
    uint64_t queen_bb = position.pieces_bb[PieceTypes::Queen] & position.colors_bb[color];
    if (queen_bb) {
      int queen_sq = get_lsb(queen_bb);
      int rank = get_rank(queen_sq);
      int bonus_rank = color == Colors::White ? 3 : 4;
      if ((color == Colors::White && rank >= bonus_rank) ||
          (color == Colors::Black && rank <= bonus_rank)) {
        mobility_bonus += 8;
      }
    }
  }

  uint64_t own_knights = position.pieces_bb[PieceTypes::Knight] & position.colors_bb[color];
  uint64_t own_bishops = position.pieces_bb[PieceTypes::Bishop] & position.colors_bb[color];
  while (own_knights) {
    int sq = pop_lsb(own_knights);
    mobility_bonus += pop_count(KnightAttacks[sq] & ~position.colors_bb[color]) / 2;
  }
  while (own_bishops) {
    int sq = pop_lsb(own_bishops);
    mobility_bonus += pop_count(get_bishop_attacks(sq, position.colors_bb[0] | position.colors_bb[1]) & ~position.colors_bb[color]) / 4;
  }

  int positional_bonus = 0;
  uint64_t home_ranks = color == Colors::White ? (Ranks[0] | Ranks[1]) : (Ranks[6] | Ranks[7]);
  uint64_t undeveloped_pieces = position.colors_bb[color] &
                                (position.pieces_bb[PieceTypes::Knight] | position.pieces_bb[PieceTypes::Bishop]) &
                                home_ranks;
  int undeveloped_count = pop_count(undeveloped_pieces);
  if (thread_info.game_ply > 10 && undeveloped_count > 0) {
    positional_bonus -= undeveloped_count * 5;
  }

  uint64_t own_rooks = position.pieces_bb[PieceTypes::Rook] & position.colors_bb[color];
  uint64_t pawns = position.pieces_bb[PieceTypes::Pawn];
  while (own_rooks) {
    int sq = pop_lsb(own_rooks);
    int f = get_file(sq);
    if (!(Files[f] & pawns)) {
      positional_bonus += 12; // Open file
    } else if (!(Files[f] & pawns & position.colors_bb[color])) {
      positional_bonus += 6; // Semi-open
    }
  }

  if (our_side) {
    bonus3 = center_control;
    bonus4 = mobility_bonus;
    bonus5 = positional_bonus;
  } else {
    bonus3 = -center_control;
    bonus4 = -mobility_bonus;
    bonus5 = -positional_bonus;
  }

  float multiplier = (800.0f + total_material / 24.0f) / 1024.0f;
  float phase_factor = 1.0f;
  switch (thread_info.phase) {
    case PhaseTypes::Opening: phase_factor = thread_info.opening_aggressiveness; break;
    case PhaseTypes::MiddleGame: phase_factor = thread_info.middlegame_aggressiveness; break;
    case PhaseTypes::LateMiddleGame: phase_factor = thread_info.late_middlegame_aggressiveness; break;
    case PhaseTypes::Endgame: phase_factor = thread_info.endgame_aggressiveness; break;
    case PhaseTypes::Sacrifice: phase_factor = thread_info.middlegame_aggressiveness * 1.1f; break;
    default: break;
  }

  if (nnue_eval > 0 && total_material > 4000) {
    multiplier *= 1.15f * phase_factor;
  } else if (nnue_eval < -100 && total_material > 4000) {
    multiplier *= 0.92f * phase_factor;
  } else {
    multiplier *= phase_factor;
  }

  if (thread_info.is_human && thread_info.search_ply < 3 && thread_info.human_noise_sigma > 0) {
    // Small opening noise proportional to configured sigma (keep bounded)
    int span = std::max(4, thread_info.human_noise_sigma / 4); // cap early influence
    int noise = (Random::dist(Random::rd) % (2 * span + 1)) - span; // uniform in [-span, span]
    nnue_eval += noise;
  }

  nnue_eval = static_cast<int>(nnue_eval * multiplier);
  return std::clamp(nnue_eval + bonus2 + bonus3 + bonus4 + bonus5, -MateScore, MateScore);
}

int correct_eval(const Position &position, ThreadInfo &thread_info, int eval) {

  eval = eval * (200 - position.halfmoves) / 200;

  int corr =
      thread_info
          .PawnCorrHist[position.color][get_corrhist_index(position.pawn_key)];

  corr +=
      thread_info
          .NonPawnCorrHist[position.color][Colors::White][get_corrhist_index(
              position.non_pawn_key[Colors::White])];
  corr +=
      thread_info
          .NonPawnCorrHist[position.color][Colors::Black][get_corrhist_index(
              position.non_pawn_key[Colors::Black])];

  return std::clamp(eval + (CorrWeight * corr / 512), -MateScore, MateScore);
}

void ss_push(Position &position, ThreadInfo &thread_info, Move move) {
  // update search stack after makemove
  if (thread_info.search_ply + 1 >= MaxSearchDepth ||
      thread_info.game_ply >= GameSize) {
    thread_data.stop = true;
    return;
  }
  ++thread_info.search_ply;

  thread_info.game_hist[thread_info.game_ply].position_key =
      position.zobrist_key;
  thread_info.game_hist[thread_info.game_ply].played_move = move;
  thread_info.game_hist[thread_info.game_ply].piece_moved =
      position.board[extract_from(move)];
  thread_info.game_hist[thread_info.game_ply].is_cap = is_cap(position, move);
  thread_info.game_hist[thread_info.game_ply].m_diff = material_eval(position);

  if (thread_info.game_ply + 1 < GameSize)
    thread_info.game_ply++;
}

void ss_pop(ThreadInfo &thread_info) {
  // associated with unmake
  thread_info.search_ply--, thread_info.game_ply--;

  thread_info.nnue_state.pop();
}

bool material_draw(
    const Position &position) { // Is there not enough material on the
                                // position for one side to win?
  for (int i : {0, 1, 6, 7, 8,
                9}) { // Do we have pawns, rooks, or queens on the position?
    if (position.material_count[i]) {
      return false;
    }
  }
  if (position.material_count[4] > 1 || position.material_count[2] > 2 ||
      (position.material_count[2] &&
       position.material_count[4])) { // Do we have three knights, two bishops,
                                      // or a bishop and knight for either side?
    return false;
  }
  if (position.material_count[5] > 1 || position.material_count[3] > 2 ||
      (position.material_count[3] &&
       position.material_count[5])) { // Do we have three knights, two bishops,
                                      // or a bishop and knight for either side?
    return false;
  }
  return true;
}

bool is_draw(const Position &position,
             ThreadInfo &thread_info) { // Detects if the position is a draw.

  uint64_t hash = position.zobrist_key;

  int halfmoves = position.halfmoves, game_ply = thread_info.game_ply;
  if (halfmoves >= 100) {
    int color = position.color;

    if (!attacks_square(position, get_king_pos(position, color), color ^ 1)) {
      return true;
    }

    MoveInfo moves;
  if (legal_movegen(position, moves.moves.data())) {
      return true;
    }
  }
  if (material_draw(position)) {
    return true;
  }
  int start_index =
      game_ply -
      4; // game_ply - 1: last played move, game_ply - 2: your last played move,
         // game_ply - 4 is the first opportunity a repetition is possible
  int end_indx = std::max(game_ply - halfmoves, 0);
  for (int i = start_index; i >= end_indx; i -= 2) {
    if (hash == thread_info.game_hist[i].position_key) {
      return true;
    }
  }
  return false;
}

int qsearch(int alpha, int beta, Position &position, ThreadInfo &thread_info,
            std::vector<TTBucket> &TT) { // Performs a quiescence search on the
                                         // given position.
  // Enforce UCI MaxDepth: if reached max depth, do not search deeper
  if (thread_info.max_depth > 0 && thread_info.search_ply >= thread_info.max_depth) {
    return correct_eval(position, thread_info, eval(position, thread_info));
  }
  if (out_of_time(thread_info)) {
    // return if out of time
    return correct_eval(
        position, thread_info,
        thread_info.nnue_state.evaluate(position.color, thread_info.phase));
  }
  
  // Check tablebase in quiescence search too
  if (thread_info.use_syzygy && tb_initialized) {
  int tb_score = probe_wdl_tb(position, thread_info);
    if (tb_score != ScoreNone) return tb_score;
  }
  
  int color = position.color;

  GameHistory *ss = &(thread_info.game_hist[thread_info.game_ply]);

  ++thread_info.nodes;

  int ply = thread_info.search_ply;

  if (ply > thread_info.seldepth) {
    thread_info.seldepth = ply;
  }
  if (ply >= MaxSearchDepth - 1) {
    return correct_eval(
        position, thread_info,
        eval(position,
             thread_info)); // if we're about to overflow stack return
  }

  uint64_t hash = position.zobrist_key;
  uint8_t phase = thread_info.phase;
  bool tt_hit;
  TTEntry &entry = probe_entry(hash, tt_hit, thread_info.searches, TT);

  int entry_type = EntryTypes::None, tt_static_eval = ScoreNone,
      tt_score = ScoreNone;
  Move tt_move = MoveNone; // Initialize TT variables and check for a hash hit

  if (tt_hit) {
    entry_type = entry.get_type();
    tt_static_eval = entry.static_eval;
    tt_score = score_from_tt(entry.score, ply);
    tt_move = entry.best_move;
  }

  if (tt_score != ScoreNone) {
    if ((entry_type == EntryTypes::Exact) ||
        (entry_type == EntryTypes::LBound && tt_score >= beta) ||
        (entry_type == EntryTypes::UBound && tt_score <= alpha)) {
      // if we get an "accurate" tt score then return
      return tt_score;
    }
  }

  uint64_t in_check =
      attacks_square(position, get_king_pos(position, color), color ^ 1);
  int best_score = ScoreNone, raised_alpha = false;
  Move best_move = MoveNone;

  int static_eval = ScoreNone;
  int raw_eval = ScoreNone;

  if (!in_check) { // If we're not in check and static eval beats beta, we can
                   // immediately return
    if (tt_static_eval == ScoreNone) {
      raw_eval = eval(position, thread_info);
      best_score = static_eval = correct_eval(position, thread_info, raw_eval);
    } else {
      raw_eval = tt_static_eval;
      best_score = static_eval = correct_eval(position, thread_info, raw_eval);
    }

    if (tt_score != ScoreNone) {
      if (entry_type == EntryTypes::Exact ||
          (entry_type == EntryTypes::UBound && tt_score < static_eval) ||
          (entry_type == EntryTypes::LBound && tt_score > static_eval)) {

        best_score = tt_score;
      }
    }

    if (best_score >= beta) {
      return best_score;
    }
    if (best_score > alpha) {
      best_move = tt_move;
      raised_alpha = true;
      alpha = best_score;
    }
  }

  MovePicker picker;
  init_picker(picker, position, -107, in_check, ss);

  if (!is_cap(position, tt_move)) {
    tt_move = MoveNone;
  }
  while (Move move =
             next_move(picker, position, thread_info, tt_move, !in_check)) {

  // Ensure we don't iterate into quiets after captures stage in qsearch
  if (picker.stage > Stages::Captures && !in_check) {
      break;
    }
    if (!is_legal(position, move)) {
      continue;
    }

    Position moved_position = position;
    make_move(moved_position, move);

    update_nnue_state(thread_info, move, position, moved_position);

    ss_push(position, thread_info, move);
    int score = -qsearch(-beta, -alpha, moved_position, thread_info, TT);
    ss_pop(thread_info);

    thread_info.phase = phase;

    if (thread_data.stop || thread_info.datagen_stop) {
      // return if we ran out of time for search
      return best_score;
    }

    if (score > best_score) {
      best_score = score;
      if (score > alpha) {
        best_move = move;
        raised_alpha = true;
        alpha = score;
      }
      if (score >= beta) { // failing high
        break;
      }
    }
  }

  if (best_score == ScoreNone) { // handle no legal moves (stalemate/checkmate)
    return Mate + ply;
  }

  // insert entries and return

  entry_type = best_score >= beta ? EntryTypes::LBound : EntryTypes::UBound;

  insert_entry(entry, hash, 0, best_move, raw_eval,
               score_to_tt(best_score, ply), entry_type, thread_info.searches);
  return best_score;
}

template <bool is_pv>
int search(int alpha, int beta, int depth, bool cutnode, Position &position,
           ThreadInfo &thread_info,
           std::vector<TTBucket> &TT) { // Performs an alpha-beta search.

  GameHistory *ss = &(thread_info.game_hist[thread_info.game_ply]);

  if (!thread_info.search_ply) {
    thread_info.current_iter = depth;
    thread_info.seldepth = 0;
    thread_info.pv.fill(MoveNone);
  }

  int ply = thread_info.search_ply, pv_index = ply * MaxSearchDepth;

  if (ply > thread_info.seldepth) {
    thread_info.seldepth = ply;
  }

  if (out_of_time(thread_info) || ply >= MaxSearchDepth - 1) {
    // check for timeout
    return correct_eval(position, thread_info, eval(position, thread_info));
  }

  if (ply && is_draw(position, thread_info)) { // Draw detection
    int draw_score = 1 - (thread_info.nodes.load() & 3);

    int material = material_eval(position);

    if (material < 0) {
      draw_score += 50;
    } else if (material > 0) {
      draw_score -= 50;
    }

    return draw_score;
    // We want to discourage draws at the root.
    // ply 0 - make a move that makes the position a draw
    // ply 1 - bonus to side, which is penalty to us

    // alternatively
    // ply 0 - we make forced move
    // ply 1 - opponent makes draw move
    // ply 2 - penalty to us*/
  }

  // Enforce UCI MaxDepth: if reached max depth, do not search deeper
  if (thread_info.max_depth > 0 && ply >= thread_info.max_depth) {
    return correct_eval(position, thread_info, eval(position, thread_info));
  }

  if (depth <= 0) {
    return qsearch(alpha, beta, position, thread_info,
                   TT); // drop into qsearch if depth is too low.
  }
  ++thread_info.nodes;

  bool root = !ply, color = position.color, raised_alpha = false;

  Move best_move = MoveNone;
  Move excluded_move = thread_info.excluded_move;

  bool singular_search = (excluded_move != MoveNone);

  if (!singular_search){
    thread_info.pv[pv_index] = MoveNone;
  }

  thread_info.excluded_move =
      MoveNone; // If we currently are in singular search, this sets it so moves
                // *after* it are not in singular search
  int score = ScoreNone;

  uint64_t hash = position.zobrist_key;
  uint8_t phase = thread_info.phase;

  int mate_distance = -Mate - ply;
  if (mate_distance <
      beta) // Mate distance pruning; if we're at depth 10 but we've already
            // found a mate in 3, there's no point searching this.
  {
    beta = mate_distance;
    if (alpha >= beta) {
      return beta;
    }
  }

  bool tt_hit;
  TTEntry &entry = probe_entry(hash, tt_hit, thread_info.searches, TT);

  int entry_type = EntryTypes::None, tt_static_eval = ScoreNone,
      tt_score = ScoreNone, tt_move = MoveNone;

  if (tt_hit && !singular_search) { // TT probe
    entry_type = entry.get_type();
    tt_static_eval = entry.static_eval;
    tt_score = score_from_tt(entry.score, ply);
    tt_move = entry.best_move;
  }

  if (tt_score != ScoreNone && !is_pv && entry.depth >= depth) {
    // If we get a useful score from the TT and it's
    // searched to at least the same depth we would
    // have searched, then we can return
    if ((entry_type == EntryTypes::Exact) ||
        (entry_type == EntryTypes::LBound && tt_score >= beta) ||
        (entry_type == EntryTypes::UBound && tt_score <= alpha)) {
      return tt_score;
    }
  }

  uint64_t in_check =
      attacks_square(position, get_king_pos(position, color), color ^ 1);

  // We can't do any eval-based pruning if in check.

  int32_t static_eval;
  int32_t raw_eval;

  if (in_check) {
    static_eval = raw_eval = ScoreNone;
  } else if (singular_search) {
    static_eval = raw_eval = ss->static_eval;
  } else {
    if (tt_static_eval == ScoreNone) {
      raw_eval = eval(position, thread_info);
      static_eval = correct_eval(position, thread_info, raw_eval);

    } else {
      raw_eval = tt_static_eval;
      static_eval = correct_eval(position, thread_info, raw_eval);
    }

    if (!tt_hit) {
      insert_entry(entry, hash, 0, MoveNone, raw_eval, ScoreNone,
                   EntryTypes::None, thread_info.searches);
    }
  }

  ss->static_eval = static_eval;

  bool improving = false;

  // Improving: Is our eval better than it was last turn? If so we can prune
  // less in certain circumstances (or prune more if it's not)

  if (ply > 1 && thread_info.game_ply >= 2 && !in_check &&
      static_eval > (ss - 2)->static_eval) {
    improving = true;
  }

  if (tt_score != ScoreNone) {
    if (entry_type == EntryTypes::Exact ||
        (entry_type == EntryTypes::UBound && tt_score < static_eval) ||
        (entry_type == EntryTypes::LBound && tt_score > static_eval)) {

      static_eval = tt_score;
    }
  }

  if (!is_pv && !in_check && !singular_search) {
    // Selective mid-node Syzygy probing (Stockfish-inspired): only at sufficient depth and low piece count
    if (thread_info.use_syzygy && tb_initialized && depth >= thread_info.syzygy_probe_depth) {
      int piece_count = pop_count(position.colors_bb[0] | position.colors_bb[1]);
      int largest = TB_LARGEST ? (int)TB_LARGEST : 7;
      if (piece_count <= std::min(thread_info.syzygy_probe_limit, largest) && !in_check && !is_pv) {
        int tb_score = probe_wdl_tb(position, thread_info);
        if (tb_score != ScoreNone) {
          // Stockfish-like: only use as bound cutoffs, do not overwrite static_eval unless decisive cutoff.
          if (tb_score >= beta) return tb_score; // Lowerbound win
          if (tb_score <= alpha) return tb_score; // Upperbound draw/loss
        }
      }
    }

    // Reverse Futility Pruning (RFP): If our position is way better than beta,
    // we're likely good to stop searching the node.

    if (depth <= RFPMaxDepth &&
        static_eval - RFPMargin * (depth - improving) >= beta) {
      return (static_eval + beta) / 2;
    }
    if (static_eval >= beta && depth >= NMPMinDepth &&
        has_non_pawn_material(position, color) &&
        thread_info.game_ply > 0 && (ss - 1)->played_move != MoveNone) {

      // Null Move Pruning (NMP): If we can give our opponent a free move and
      // still beat beta on a reduced search, we can prune the node.

      Position temp_pos = position;
      make_move(temp_pos, MoveNone);

      ss_push(position, thread_info, MoveNone);

      int R = NMPBase + depth / NMPDepthDiv +
              std::min(3, (static_eval - beta) / NMPEvalDiv);
      score = -search<false>(-alpha - 1, -alpha, depth - R, !cutnode, temp_pos,
                             thread_info, TT);

      thread_info.search_ply--, thread_info.game_ply--;
      // we don't call ss_pop because the nnue state was never pushed

      if (score >= beta) {
        if (score > MateScore) {
          score = beta;
        }
        return score;
      }
    }
  }

  if ((is_pv || cutnode) && tt_move == MoveNone && depth > IIRMinDepth) {
    // Internal Iterative Reduction: If we are in a PV node and have no TT move,
    // reduce the depth.
    depth--;
  }

  int p_beta = beta + 250;
  if (depth >= 5 && abs(beta) < MateScore &&
      (!tt_hit || entry.depth + 4 <= depth || tt_score >= p_beta)) {

    int threshold = p_beta - static_eval;
    MovePicker probcut_p;
    init_picker(probcut_p, position, threshold, in_check, ss);
    Move p_tt_move =
        (tt_move != MoveNone && SEE(position, tt_move, threshold) ? tt_move
                                                                  : MoveNone);

    while (Move move = next_move(probcut_p, position, thread_info, p_tt_move, true)) {

      if (probcut_p.stage > Stages::Captures) {
        break;
      }
      if (move == excluded_move || !is_legal(position, move)) {
        continue;
      }

      Position moved_position = position;
      make_move(moved_position, move);
      update_nnue_state(thread_info, move, position, moved_position);
      ss_push(position, thread_info, move);

      int score =
          -qsearch(-p_beta, -p_beta + 1, moved_position, thread_info, TT);
      if (score >= p_beta) {
        score = -search<is_pv>(-p_beta, -p_beta + 1, depth - 4, false,
                               moved_position, thread_info, TT);
      }

      ss_pop(thread_info);
      thread_info.phase = phase;

      if (score >= p_beta) {
        return score;
      }
    }
  }

  Move quiets[64];
  int num_quiets = 0;
  Move captures[64];
  int num_captures = 0;
  thread_info.KillerMoves[ply + 1] = MoveNone;

  MovePicker picker;
  init_picker(picker, position, -107, in_check, ss);

  int best_score = ScoreNone, moves_played = 0; // Generate and score moves
  bool is_capture = false, skip = false;
  
  // Materyal feda analizi (SEE tabanlı)
  // İlk iki katmanda (ply < 2) ve sacrifice_lookahead > 0 ise potansiyel fedaları değerlendir.
  // Tetikleme ply sınırı artık doğrudan sacrifice_lookahead değeri (0..6) ile aynı
  // LookAhead = N  => ilk N ply (0..N-1) içinde feda analizi aktif
  // (LookAhead=0 kapalı, 1 sadece kök, 2 kök+1, ...)
  if (thread_info.search_ply < thread_info.sacrifice_lookahead && thread_info.sacrifice_lookahead > 0 && !in_check) {
    std::array<Move, ListSize> moves;
    uint64_t checkers_local = attacks_square(position, get_king_pos(position, color), color ^ 1);
    int nmoves_local = movegen(position, moves.data(), checkers_local, Generate::GenAll);
  // Statik lookahead sınırı: doğrudan kullanıcı ayarı (0..6)
  int lookahead_cap = std::clamp(thread_info.sacrifice_lookahead, 0, 6);

    for (int i = 0; i < nmoves_local; i++) {
      Move m = moves[i];
      if (!is_legal(position, m)) continue;

      // SEE filtresi: Kayba açık / şüpheli değişimler (SEE başarısız) veya doğrudan capture ise değerlendir
      bool isCapture = is_cap(position, m);
      bool losing_exchange = !SEE(position, m, 0); // 0 eşiği: eşit ya da kârlı değilse potansiyel feda
      if (!isCapture && !losing_exchange) continue;

      Position test_position = position;
      make_move(test_position, m);

      int sacrifice_score = analyze_sacrifice(test_position, thread_info, lookahead_cap, 0, position.color);
      if (sacrifice_score <= 0) continue; // sadece pozitif potansiyel

      // Derinlik uzatması: agresifliğe göre 1..3 ply
      int sac_extension = 1 + (thread_info.sacrifice_lookahead_aggressiveness >= 130 ? 2 : (thread_info.sacrifice_lookahead_aggressiveness >= 90 ? 1 : 0));
      sac_extension = std::clamp(sac_extension, 1, 3);

      Position moved_position = position;
      make_move(moved_position, m);
      update_nnue_state(thread_info, m, position, moved_position);
      ss_push(position, thread_info, m);

      // Önce hızlı null pencere test (derinlik-1)
      int probe_score = -search<false>(-alpha - 1, -alpha, std::max(1, depth - 1), false, moved_position, thread_info, TT);
      int score;
      if (probe_score > alpha && abs(probe_score) < MateScore) {
        // Geniş pencere yerine daha derin tam arama (derinlik + uzatma)
        int extended_depth = std::min(depth + sac_extension, 126);
        score = -search<true>(-beta, -alpha, extended_depth, false, moved_position, thread_info, TT);
      } else {
        score = probe_score;
      }

      ss_pop(thread_info);
      thread_info.phase = phase;

      if (abs(score) < MateScore && score > alpha) {
        // Dinamik bonus (oyun fazına göre) + agresiflik çarpanı
        float phase_bonus = 0.0f;
        int move_number_local = (thread_info.game_ply / 2) + 1;
        if (move_number_local < 10) phase_bonus = thread_info.opening_aggressiveness * 15.0f;
        else if (move_number_local < 25) phase_bonus = thread_info.middlegame_aggressiveness * 20.0f;
        else if (move_number_local < 40) phase_bonus = thread_info.late_middlegame_aggressiveness * 25.0f;
        else phase_bonus = thread_info.endgame_aggressiveness * 15.0f;

        float aggr_factor_local = thread_info.sacrifice_lookahead_aggressiveness / 100.0f;
        int bonus = static_cast<int>(phase_bonus * aggr_factor_local);
        score += bonus;

        // HistoryScores güncelle: küçük pozitif takviye (sacrifice_score ölçekli)
        int piece_from = position.board[extract_from(m)];
        int to_sq = extract_to(m);
        int hist_bonus = std::clamp(16 + sacrifice_score / 8, 8, 128); // sınırlı güç
        if (thread_info.attack_mode) {
          hist_bonus = hist_bonus * 6 / 5 + 6; // %20 güçlendir + küçük ek
          hist_bonus = std::min(hist_bonus, 192);
        }
        update_history(thread_info.HistoryScores[piece_from][to_sq], hist_bonus);

        if (score > best_score) {
          best_score = score;
          if (score > alpha) {
            alpha = score;
            best_move = m;
            raised_alpha = true;
            if (is_pv) thread_info.pv[pv_index] = m;
            if (score >= beta) break;
          }
        }
      }
      if (thread_data.stop || thread_info.datagen_stop) break;
    }
  }

  while (Move move = next_move(picker, position, thread_info, tt_move, skip)) {

    if (root) {
      bool pv_skip = false;
      for (int i = 0; i < thread_info.multipv_index; i++) {
        if (thread_info.best_moves[i] == move) {
          pv_skip = true;
          break;
        }
      }
      if (pv_skip) {
        continue;
      }
    }

    if (move == excluded_move) {
      continue;
    }
    if (!is_legal(position, move)) {
      continue;
    }

    uint64_t curr_nodes = thread_info.nodes.load();

    int hist_score =
        thread_info.HistoryScores[position.board[extract_from(move)]]
                                 [extract_to(move)];

    is_capture = is_cap(position, move);
    if (!is_capture && !is_pv && best_score > -MateScore) {

      // Late Move Pruning (LMP): If we've searched enough moves, we can skip
      // the rest.

      if (depth < LMPDepth &&
          moves_played >= LMPBase + depth * depth / (2 - improving)) {
        skip = true;
      }

      // Futility Pruning (FP): If we're far worse than alpha and our move isn't
      // a good capture, we can skip the rest.

      if (!in_check && depth < FPDepth && picker.stage > Stages::Captures) {
        int fp_margin = FPMargin1 + FPMargin2 * depth;
        if (thread_info.attack_mode) fp_margin += 60; // saldırıda daha fazla genişlet
        if (static_eval + fp_margin < alpha) {
        skip = true;
        }
      }

      if (!is_pv && !is_capture && depth < 4 && hist_score < -4096 * depth) {
        skip = true;
      }
    }

    if (!root && best_score > -MateScore && depth < SeePruningDepth) {

      int margin =
          is_capture ? SeePruningQuietMargin : (depth * SeePruningNoisyMargin);

      if (!SEE(position, move, depth * margin)) {
        // SEE pruning: if we are hanging material, prune under certain
        // conditions.
        continue;
      }
    }

    int extension = 0;

    // Singular Extensions (SE): If a search finds that the TT move is way
    // better than all other moves, extend it under certain conditions.

    if (!root && ply < thread_info.current_iter * 2) {
      if (!singular_search && depth >= SEDepth && move == tt_move &&
          abs(entry.score) < MateScore && entry.depth >= depth - 3 &&
          entry_type != EntryTypes::UBound) {

        int sBeta = entry.score - depth;
        thread_info.excluded_move = move;
        int sScore = search<false>(sBeta - 1, sBeta, (depth - 1) / 2, cutnode,
                                   position, thread_info, TT);

        if (sScore < sBeta) {
          if (!is_pv && sScore + SEDoubleExtMargin < sBeta &&
              ply < thread_info.current_iter) {

            // In some cases we can even double extend
            extension = 2 + (!is_capture && sScore < sBeta - 125);
          } else {
            extension = 1;
          }
        } else if (sBeta >= beta) {
          // Multicut: If there was another move that beat beta, it's a sign
          // that we'll probably beat beta with a full search too.

          return sBeta;
        } else if (cutnode) {
          extension = -1;
        }
      }
    }

    Position moved_position = position;
    make_move(moved_position, move);

    update_nnue_state(thread_info, move, position, moved_position);

    ss_push(position, thread_info, move);

    bool full_search = false;
    int newdepth = std::min(depth - 1 + extension, 126);

    // Late Move Reductions (LMR): Moves ordered later in search and at high
    // depths can be searched to a lesser depth than normal. If the reduced
    // search beats alpha, we'll have to search again, but most moves don't,
    // making this technique more than worth it.
    // If that beats alpha, we search at normal depth with null window
    // If that also beats alpha, we search at normal depth with full window.

    if (depth >= LMRMinDepth && moves_played > is_pv) {
      int R = LMRTable[depth][moves_played];
      if (is_capture) {
        // Captures get LMRd less because they're the most likely moves to beat
        // alpha/beta
        R /= 2;
      } else {
        R -= hist_score / 10000;
      }

      // Increase reduction if not in pv
      R -= is_pv;

      R -= (tt_hit && entry.depth >= depth);

      // Increase reduction if not improving
      R += !improving;

      R += cutnode;

      R -= (attacks_square(moved_position, get_king_pos(position, color ^ 1), color) != 0);


      // Attack mode'da daha sığ azalt: agresiflik arttırma
      if (thread_info.attack_mode && R > 0) {
        R = std::max(0, R - 1);
      }

      // Clamp reduction so we don't immediately go into qsearch
      R = std::clamp(R, 0, newdepth - 1);

      // Reduced search, reduced window
      score = -search<false>(-alpha - 1, -alpha, newdepth - R, true,
                             moved_position, thread_info, TT);
      if (score > alpha) {
        full_search = R > 0;
        newdepth += (score > (best_score + 60 + newdepth * 2));
        newdepth -= (score < best_score + newdepth && !root);
      }
    } else {
      full_search = moves_played || !is_pv;
    }
    if (full_search) {
      // Full search, null window
      score = -search<false>(-alpha - 1, -alpha, newdepth, !cutnode,
                             moved_position, thread_info, TT);
    }
    if ((score > alpha || !moves_played) && is_pv) {
      // Full search, full window
      score = -search<true>(-beta, -alpha, newdepth, false, moved_position,
                            thread_info, TT);
    }

    ss_pop(thread_info);
    thread_info.phase = phase;

    if (thread_data.stop || thread_info.datagen_stop) {
      // return if we ran out of time for search
      return best_score;
    }

    if (root) {
      find_root_move(thread_info, move)->nodes +=
          (thread_info.nodes.load() - curr_nodes);
    }

    if (score > best_score) {
      best_score = score;

      if (score > alpha) {
        best_move = move;
        raised_alpha = true;
        alpha = score;

        if (score >= beta) {
          break;
        }

        else {

          thread_info.pv[pv_index] = best_move;
          for (int n = 0; n < MaxSearchDepth + ply + 1; n++) {
            thread_info.pv[pv_index + 1 + n] =
                thread_info.pv[pv_index + MaxSearchDepth + n];
          }
        }
      }
    }

    if (is_capture) {
      if (num_captures < 64)
        captures[num_captures++] = move;
    } else {
      if (num_quiets < 64)
        quiets[num_quiets++] = move;
    }

    moves_played++;
  }

  if (root) {
    if (best_move != MoveNone) {
      thread_info.best_moves[thread_info.multipv_index] = best_move;
    }
    thread_info.best_scores[thread_info.multipv_index] = best_score;
  }

  if (best_score >= beta) {

    int piece = position.board[extract_from(best_move)],
        sq = extract_to(best_move);

    int bonus = std::min((int)HistBonus * (depth - 1 + (best_score > beta + 125)), (int)HistMax);

    // Update history scores and the killer move.

    if (is_capture) {

      update_history(thread_info.CapHistScores[piece][sq], bonus);

    } else {

      Move their_last = MoveNone;
      int their_piece = Pieces::Blank;
      Move our_last = MoveNone;
      int our_piece = Pieces::Blank;
      Move ply4_last = MoveNone;
      int ply4_piece = Pieces::Blank;

      if (thread_info.game_ply >= 1) {
        their_last = extract_to((ss - 1)->played_move);
        their_piece = (ss - 1)->piece_moved;
      }
      if (thread_info.game_ply >= 2) {
        our_last = extract_to((ss - 2)->played_move);
        our_piece = (ss - 2)->piece_moved;
      }
      if (thread_info.game_ply >= 4) {
        ply4_last = extract_to((ss - 4)->played_move);
        ply4_piece = (ss - 4)->piece_moved;
      }

      for (int i = 0; i < num_quiets; i++) {

        // Every quiet move that *didn't* raise beta gets its history score
        // reduced

        Move move = quiets[i];

        int piece_m = position.board[extract_from(move)],
            sq_m = extract_to(move);

        update_history(thread_info.HistoryScores[piece_m][sq_m], -bonus);

        if (their_last != MoveNone) {
          update_history(
              thread_info.ContHistScores[their_piece][their_last][piece_m][sq_m],
              -bonus);
        }

        if (our_last != MoveNone) {
          update_history(
              thread_info.ContHistScores[our_piece][our_last][piece_m][sq_m],
              -bonus);
        }

        if (ply4_last != MoveNone) {
          update_history(
              thread_info.ContHistScores[ply4_piece][ply4_last][piece_m][sq_m],
              -bonus / 2);
        }
      }

      update_history(thread_info.HistoryScores[piece][sq], bonus);

      if (their_last != MoveNone) {
        update_history(
            thread_info.ContHistScores[their_piece][their_last][piece][sq],
            bonus);
      }
      if (our_last != MoveNone) {
        update_history(
            thread_info.ContHistScores[our_piece][our_last][piece][sq],
            bonus);
      }
      if (ply4_last != MoveNone) {
        update_history(
            thread_info.ContHistScores[ply4_piece][ply4_last][piece][sq],
            bonus / 2);
      }

      thread_info.KillerMoves[ply] = best_move;
    }

    for (int i = 0; i < num_captures; i++) {
      Move move = captures[i];

      int piece_m = position.board[extract_from(move)], sq_m = extract_to(move);

      update_history(thread_info.CapHistScores[piece_m][sq_m], -bonus);
    }
  }

  if (best_score == ScoreNone) { // handle no legal moves (stalemate/checkmate)
    return singular_search ? alpha : in_check ? (Mate + ply) : 0;
  }

  entry_type = best_score >= beta ? EntryTypes::LBound
               : raised_alpha     ? EntryTypes::Exact
                                  : EntryTypes::UBound;

  bool best_capture = is_cap(position, best_move);

  if (!in_check && (!best_move || !best_capture) &&
      !(best_score >= beta && best_score <= ss->static_eval) &&
      !(!best_move && best_score >= ss->static_eval)) {

    int bonus = std::clamp((best_score - ss->static_eval) * depth / 8, -256, 256);

    update_corrhist(
        thread_info.PawnCorrHist[color][get_corrhist_index(position.pawn_key)],
        bonus);
    update_corrhist(
        thread_info.NonPawnCorrHist[color][Colors::White][get_corrhist_index(
            position.non_pawn_key[Colors::White])],
        bonus);
    update_corrhist(
        thread_info.NonPawnCorrHist[color][Colors::Black][get_corrhist_index(
            position.non_pawn_key[Colors::Black])],
        bonus);
  }

  // Add the search results to the TT, accounting for mate scores
  if (!singular_search) {
    insert_entry(entry, hash, depth, best_move, raw_eval,
                 score_to_tt(best_score, ply), entry_type,
                 thread_info.searches);
  }

  return best_score;
}

void print_pv(Position &position, ThreadInfo &thread_info) {
  Position temp_pos = position;

  int indx = 0;

  while (thread_info.pv[indx] != MoveNone) {

    if (indx == 3 && thread_info.is_human) {
      thread_info.pv_material[thread_info.multipv_index] =
          -material_eval(temp_pos);
    }

    Move best_move = thread_info.pv[indx];

    // Verify that the pv move is possible and legal by generating moves

    MoveInfo moves;
  int movelen = legal_movegen(temp_pos, moves.moves.data());

    bool found_move = false;

    for (int i = 0; i < movelen; i++) {
      if (moves.moves[i] == best_move) {
        found_move = true;
        break;
      }
    }

    if (!found_move) {
      break;
    }

    printf("%s ", internal_to_uci(temp_pos, best_move).c_str());

    make_move(temp_pos, best_move);

    indx++;
  }

  printf("\n");
}

void iterative_deepen(
    Position &position, ThreadInfo &thread_info,
    std::vector<TTBucket> &TT) { // Performs an iterative deepening search.

  thread_info.original_opt = thread_info.opt_time;
  thread_info.datagen_stop = false;
  
  // Determine the initial phase based on material and game progress
  int material = total_mat(position);
  int move_number = (thread_info.game_ply / 2) + 1;
  
  // Phase detection logic
  if (move_number < 15 && material > 4500) {
    thread_info.phase = PhaseTypes::Opening;
  }
  else if (move_number >= 15 && move_number < 30 && material > 4000) {
    thread_info.phase = PhaseTypes::MiddleGame;
  }
  else if (material > 3000 && material <= 4000) {
    thread_info.phase = PhaseTypes::LateMiddleGame;
  }
  else if (material <= 3000) {
    thread_info.phase = PhaseTypes::Endgame;
  }
  
  thread_info.nnue_state.reset_nnue(position, thread_info.phase);
  calculate(position);
  thread_info.nodes.store(0);
  thread_info.time_checks = 0;
  thread_info.search_ply = 0; // reset all relevant thread_info
  thread_info.excluded_move = MoveNone;
  thread_info.best_moves = {0};
  thread_info.best_scores = {ScoreNone, ScoreNone, ScoreNone, ScoreNone, ScoreNone};
  thread_info.KillerMoves.fill(MoveNone);
  // Root Syzygy DTZ/WDL ranking integration
  bool tb_decisive_shortcut = false;
  if (thread_info.use_syzygy && tb_initialized) {
    // Skip probing if any castling rights remain (tablebases defined only for no castling rights)
    bool any_castling = false;
    for (int c=0;c<2;c++) for (int s=0;s<2;s++) if (position.castling_squares[c][s] != SquareNone) any_castling = true;
    if (any_castling) goto skip_tb_root;
    // Count pieces quickly
    int piece_count = 0; for (int i=0;i<64;i++) if (position.board[i]) piece_count++;
    if (piece_count && TB_LARGEST && piece_count <= (int)TB_LARGEST) {
      uint64_t white=0, black=0, kings=0, queens=0, rooks=0, bishops=0, knights=0, pawns=0;
      for (int sq=0; sq<64; ++sq) {
        int pc = position.board[sq]; if(!pc) continue; int pt = get_piece_type(pc); int c = get_color(pc); uint64_t bb=1ULL<<sq;
        if (c==Colors::White) white|=bb; else black|=bb;
        switch(pt){case PieceTypes::King: kings|=bb; break; case PieceTypes::Queen: queens|=bb; break; case PieceTypes::Rook: rooks|=bb; break; case PieceTypes::Bishop: bishops|=bb; break; case PieceTypes::Knight: knights|=bb; break; case PieceTypes::Pawn: pawns|=bb; break; default: break;}
      }
      unsigned ep = position.ep_square<64 ? (position.ep_square%8)+1 : 0;
  unsigned rule50 = thread_info.syzygy_50_move_rule ? position.halfmoves : 0;
      unsigned castling = 0; // TB requires 0 for castling in probe functions used here
      TbRootMoves tbMoves{}; int ok = 0;
      // Try DTZ first
      ok = tb_probe_root_dtz(kings|(queens|rooks|bishops|knights|pawns)&white,
                             kings|(queens|rooks|bishops|knights|pawns)&black,
                             kings, queens, rooks, bishops, knights, pawns,
                             rule50, castling, ep, position.color==Colors::White,
                             false, true, &tbMoves);
      if (!ok) {
        ok = tb_probe_root_wdl(kings|(queens|rooks|bishops|knights|pawns)&white,
                               kings|(queens|rooks|bishops|knights|pawns)&black,
                               kings, queens, rooks, bishops, knights, pawns,
                               rule50, castling, ep, position.color==Colors::White,
                               true, &tbMoves);
      }
      if (ok) {
        // Convert TbRootMoves into root_moves ordering (will rebuild later root_moves vector)
        std::vector<std::pair<Move,int>> tbOrdered;
        for (unsigned i=0; i<tbMoves.size; ++i) {
          TbMove tm = tbMoves.moves[i].move; int from = TB_MOVE_FROM(tm); int to = TB_MOVE_TO(tm); int promo = TB_MOVE_PROMOTES(tm);
          Move m = promo ? pack_move_promo(from,to,promo-1) : pack_move(from,to,MoveTypes::Normal);
          int wdl = TB_GET_WDL(tbMoves.moves[i].tbScore);
          int mapped;
          switch(wdl){
            case TB_WIN: mapped = TB_WIN_SCORE; break;
            case TB_CURSED_WIN: mapped = TB_CURSED_WIN_SCORE; break;
            case TB_DRAW: mapped = TB_DRAW_SCORE; break;
            case TB_BLESSED_LOSS: mapped = TB_BLESSED_LOSS_SCORE; break;
            case TB_LOSS: mapped = TB_LOSS_SCORE; break;
            default: mapped = 0; break;
          }
          tbOrdered.emplace_back(m, mapped);
        }
        // Heuristic: if any winning WDL present and none losing for side to move, can shortcut.
        bool hasWin=false, hasLoss=false; for (auto &p: tbOrdered){ if (p.second > MateScore-1000) hasWin=true; if (p.second < -MateScore+1000) hasLoss=true; }
        if (hasWin && !hasLoss && !thread_info.infinite_search) {
          // choose first winning move
            thread_info.best_moves[0] = tbOrdered[0].first; thread_info.ponder_move = MoveNone; tb_decisive_shortcut = true;
            // verbose TB shortcut suppressed
            printf("bestmove %s\n", internal_to_uci(position, tbOrdered[0].first).c_str());
            return;
        }
        // Rebuild root_moves with TB ordering at front
        thread_info.root_moves.clear();
        for (auto &p: tbOrdered) thread_info.root_moves.push_back({p.first,0});
        // If not decisive, fall through to search (ordering improved)
  // verbose ordering info suppressed
      }
    }
  }
skip_tb_root: ;

  // Prepare root moves
  thread_info.root_moves.reserve(ListSize);
  thread_info.root_moves.clear();
  {
    std::array<Move, ListSize> raw_root_moves;
  int nmoves = legal_movegen(position, raw_root_moves.data());
    for (int i = 0; i < nmoves; i++) {
      thread_info.root_moves.push_back({raw_root_moves[i], 0});
    }
  }

  Move prev_best = MoveNone;
  int alpha = ScoreNone, beta = -ScoreNone;
  int bm_stability = 0;

  int target_depth = std::clamp(thread_info.max_iter_depth, 1, MaxSearchDepth);
  int last_completed_depth = 0;

  auto update_phase = [&](ThreadInfo &ti, Position &pos) {
    // Sadece kök thread (id 0) faz kararını verir
    if (ti.thread_id != 0) return;

    int total_material = total_mat(pos);
    int root_eval = ti.best_scores[0]; // cp
    ti.prev_root_eval = ti.last_root_eval;
    ti.last_root_eval = root_eval;
    ti.root_completed_depth = last_completed_depth;

    // Attack mode histerezis (sacrifice replacement)
    if (!ti.attack_mode) {
      if (last_completed_depth >= 8 && root_eval >= ti.sacrifice_enter_cp && total_material >= 3200) {
        // İki ardışık şart gerek: geçmişte de üstte miydi?
        if (ti.prev_root_eval >= ti.sacrifice_enter_cp) {
          ti.attack_mode = true;
        }
      }
    } else {
      bool drop = (ti.prev_root_eval - root_eval) >= ti.sacrifice_drop_threshold;
      if (root_eval <= ti.sacrifice_exit_cp || total_material < ti.endgame_material || drop) {
        ti.attack_mode = false;
      }
    }

    // Hedef faz belirleme
    uint8_t desired_phase = ti.phase; // default aynı kalsın

    // Opening'ten çıkış: ply + gelişmiş hafif taş kriterleri (basit: game_ply >= opening_min_ply)
    if (ti.phase == PhaseTypes::Opening && ti.game_ply >= ti.opening_min_ply) {
      desired_phase = PhaseTypes::MiddleGame;
    }

    // Endgame geri dönüşleri (tamponlar)
    if (ti.phase == PhaseTypes::Endgame) {
      if (total_material > ti.end_recover_material && total_material > ti.endgame_material) {
        desired_phase = PhaseTypes::LateMiddleGame; // bir seviye yukarı
      }
    }
    // LateMiddleGame geri Middle
    else if (ti.phase == PhaseTypes::LateMiddleGame) {
      if (total_material > ti.mid_recover_material) {
        desired_phase = PhaseTypes::MiddleGame;
      }
    }

    // İleri geçişler
    if (total_material <= ti.endgame_material) {
      desired_phase = PhaseTypes::Endgame;
    } else if (total_material <= ti.late_phase_material) {
      // Endgame değilse LateMiddleGame'e
      if (desired_phase != PhaseTypes::Endgame)
        desired_phase = PhaseTypes::LateMiddleGame;
    } else {
      // Yüksek materyalde Opening/MG seçimi
      if (ti.game_ply < ti.opening_min_ply)
        desired_phase = PhaseTypes::Opening;
      else
        desired_phase = PhaseTypes::MiddleGame;
    }

    // Attack mode aktifse, ayrı bir Sacrifice fazına map etme: sadece farklı ağ gereksinimi varsa
    // Burada Sacrifice fazını fiziksel ağ değiştirmeyecek şekilde devre dışı bırakıyoruz (opsiyonel)
    // İstenirse: desired_phase = PhaseTypes::Sacrifice; mantığı eklenebilir.

    // Histerezis doğrulama
    if (desired_phase != ti.phase) {
      ti.phase_hit_counts[desired_phase]++;
      // diğer faz aday sayaçlarını sıfırla (sadece hedefi arttır)
      for (size_t i = 0; i < ti.phase_hit_counts.size(); ++i) {
        if (i != desired_phase) ti.phase_hit_counts[i] = 0;
      }
      if (ti.phase_hit_counts[desired_phase] >= ti.phase_confirm_hits) {
        uint8_t old_phase = ti.phase;
        ti.phase = desired_phase;
        // Fiziksel ağ gerçekten değişiyor mu kontrolü
        auto physical_net = [](uint8_t ph) {
          switch (ph) {
            case PhaseTypes::Opening:
            case PhaseTypes::MiddleGame: return 0; // Net A
            case PhaseTypes::LateMiddleGame: return 1; // Net B
            case PhaseTypes::Endgame: return 2; // Net C
            case PhaseTypes::Sacrifice: return 2; // Sacrifice -> Endgame ağına bağladık
            default: return 0;
          }
        };
        if (physical_net(old_phase) != physical_net(ti.phase)) {
          ti.nnue_state.reset_nnue(pos, ti.phase);
        }
      }
    } else {
      // Faz değişmedi, sayaç sıfırla
      for (auto &c : ti.phase_hit_counts) c = 0;
    }
  };
  int real_multi_pv = std::min<int>(thread_info.multipv, (int)thread_info.root_moves.size());
  
  for (int depth = 1;; ++depth) {
    if (thread_data.stop) {
      break;
    }
    if (thread_info.infinite_search && depth > MaxSearchDepth) {
      depth = MaxSearchDepth;
    }
    if (!thread_info.infinite_search && depth > target_depth) {
      break;
    }

  real_multi_pv = std::min<int>(thread_info.multipv, (int)thread_info.root_moves.size());
    
    for (thread_info.multipv_index = 0; 
         thread_info.multipv_index < real_multi_pv;
         thread_info.multipv_index++) {

      int temp_depth = depth;

      int score, delta = AspStartWindow;

      score =
          search<true>(alpha, beta, depth, false, position, thread_info, TT);

      // Aspiration Windows: We search the position with a narrow window around
      // the last search score in order to get cutoffs faster. If our search
      // lands outside the bounds, expand them and try again.

      while (score <= alpha || score >= beta || thread_data.stop ||
             thread_info.datagen_stop) {

        if (thread_data.stop || thread_info.datagen_stop) {
          goto finish;
        }

        if (thread_info.thread_id == 0 && !thread_info.doing_datagen &&
            !(thread_info.is_human && thread_info.multipv_index)) {
          std::string bound_string;
          if (score >= beta) {
            bound_string = "lowerbound";
          } else {
            bound_string = "upperbound";
          }

          uint64_t nodes = thread_info.nodes.load();
          for (auto &td : thread_data.thread_infos) {
            nodes += td.nodes.load();
          }
          int64_t search_time = time_elapsed(thread_info.start_time);
          int64_t nps = search_time
                            ? static_cast<int64_t>(nodes) * 1000 / search_time
                            : 123456789;

          Move move = score <= alpha
                          ? prev_best
                          : thread_info.best_moves[thread_info.multipv_index];

          if (abs(score) <= MateScore) {
            printf("info multipv %i depth %i seldepth %i score cp %i %s nodes "
                   "%" PRIu64 " nps %" PRIi64 " time %" PRIi64 " pv %s\n",
                   thread_info.multipv_index + 1, depth, thread_info.seldepth,
                   score * 100 / NormalizationFactor, bound_string.c_str(), nodes,
                   nps, search_time, internal_to_uci(position, move).c_str());
          } else if (score > MateScore) {
            int dist = (MateScore - score) / 2;
            printf("info multipv %i depth %i seldepth %i score mate %i %s nodes "
                   "%" PRIu64 " nps %" PRIi64 " time %" PRIi64 " pv %s\n",
                   thread_info.multipv_index + 1, depth, thread_info.seldepth,
                   dist, bound_string.c_str(), nodes, nps,
                   search_time, internal_to_uci(position, move).c_str());
          } else {
            int dist = (Mate - score) / 2;
            printf("info multipv %i depth %i seldepth %i score mate %i %s nodes "
                   "%" PRIu64 " nps %" PRIi64 " time %" PRIi64 " pv %s\n",
                   thread_info.multipv_index + 1, depth, thread_info.seldepth,
                   dist, bound_string.c_str(), nodes, nps,
                   search_time, internal_to_uci(position, move).c_str());
          }
        }

        if (score <= alpha) {
          beta = (alpha + beta) / 2;
          alpha -= delta;
          temp_depth = depth;
        } else if (score >= beta) {
          beta += delta;
          temp_depth = std::max(temp_depth - 1, 1);
        }
        delta += delta / 3;

        score = search<true>(alpha, beta, temp_depth, false, position,
                             thread_info, TT);
      }

      if (score == ScoreNone) {
        break;
      }

      std::string eval_string;

      if (abs(score) <= MateScore) {
        eval_string = "cp " + std::to_string(score * 100 / NormalizationFactor);
      } else if (score > MateScore) {
        int dist = (MateScore - score) / 2;
        eval_string = "mate " + std::to_string(dist);
      } else {
        int dist = (Mate - score) / 2;
        eval_string = "mate " + std::to_string(dist);
      }

      thread_info.best_moves[thread_info.multipv_index] = thread_info.pv[0];

  if (thread_info.thread_id == 0) {

        uint64_t nodes = thread_info.nodes.load();

        for (auto &td : thread_data.thread_infos) {
          nodes += td.nodes.load();
        }

        int64_t search_time = time_elapsed(thread_info.start_time);
        int64_t nps;
        if (search_time) {
          nps = static_cast<int64_t>(nodes) * 1000 / search_time;
        } else {
          int wezly = 10000000;
          wezly += (wezly / 7);
          nps = wezly;
        }

        if (!thread_info.doing_datagen /*&&
            !(thread_info.is_human && thread_info.multipv_index)*/) {
          printf("info multipv %i depth %i seldepth %i score %s nodes %" PRIu64
                 " nps %" PRIi64 " time %" PRIi64 " pv ",
                 thread_info.multipv_index + 1, depth, thread_info.seldepth,
                 eval_string.c_str(), nodes, nps, search_time);
          print_pv(position, thread_info);
        }

        else {
          thread_info.best_scores[0] = score * 100 / NormalizationFactor;
        }

        if (static_cast<uint64_t>(search_time) > thread_info.opt_time ||
            nodes > thread_info.opt_nodes_searched) {

          if (thread_info.doing_datagen) {
            thread_info.datagen_stop = true;
          } else {
            thread_data.stop = true;
          }
        }

        else if (thread_info.multipv == 1 && depth > 6) {
          if (thread_info.best_moves[0] == prev_best) {
            bm_stability = std::min(bm_stability + 1, 8);
            thread_info.stability_counter++;
            thread_info.best_move_stable = (thread_info.stability_counter >= 3);
          } else {
            bm_stability = 0;
            thread_info.stability_counter = 0;
            thread_info.best_move_stable = false;
            thread_info.previous_best_move = prev_best;
          }

          adjust_soft_limit(
              thread_info,
              find_root_move(thread_info, thread_info.best_moves[0])->nodes,
              bm_stability);
        }

  // Eski tek-seferlik derinlik 6 faz geçişleri kaldırıldı
      }

      if (thread_data.stop || thread_info.datagen_stop) {
        goto finish;
      }

      prev_best = thread_info.best_moves[0];

  if (depth > 6 && thread_info.multipv_index == 0) {
        alpha = score - 20, beta = score + 20;
      } else {
        alpha = ScoreNone, beta = -ScoreNone;
      }
    }
    
    last_completed_depth = depth;
  }

  // Yeni faz güncelleme sistemi
  update_phase(thread_info, position);

finish:
  // wait for all threads to finish searching
  if (thread_info.thread_id == 0 && !thread_info.doing_datagen) {
    thread_data.stop = true;
  }
  search_end_barrier.arrive_and_wait();

  auto validate_ponder_move = [&](const Position &root_position, Move best_move,
                                  Move ponder_candidate) -> Move {
    if (ponder_candidate == MoveNone || best_move == MoveNone) {
      return MoveNone;
    }

    std::array<Move, ListSize> root_legal{};
    int root_count = legal_movegen(root_position, root_legal.data());
    bool best_is_legal = false;
    for (int i = 0; i < root_count; ++i) {
      if (root_legal[i] == best_move) {
        best_is_legal = true;
        break;
      }
    }
    if (!best_is_legal) {
      return MoveNone;
    }

    Position ponder_position = root_position;
    make_move(ponder_position, best_move);

    std::array<Move, ListSize> response_legal{};
    int response_count = legal_movegen(ponder_position, response_legal.data());
    for (int i = 0; i < response_count; ++i) {
      if (response_legal[i] == ponder_candidate) {
        return ponder_candidate;
      }
    }

    return MoveNone;
  };

  // Enhanced ponder move calculation - always try to set ponder move from PV
  if (thread_info.thread_id == 0) {
    // Try to get ponder move from principal variation
    if (thread_info.pv[0] != MoveNone && thread_info.pv[1] != MoveNone) {
      thread_info.ponder_move = thread_info.pv[1];
    } else {
      // Fallback: try to predict opponent's likely response
      Position temp_pos = position;
      if (thread_info.best_moves[0] != MoveNone) {
        make_move(temp_pos, thread_info.best_moves[0]);
        
        // Simple response prediction - choose most natural reply
        std::array<Move, ListSize> responses;
  int num_responses = legal_movegen(temp_pos, responses.data());
        
        if (num_responses > 0) {
          // Prefer captures or center control moves as ponder candidates
          Move best_response = responses[0];
          int best_score = -1000;
          
          for (int i = 0; i < std::min(num_responses, 5); i++) {
            int score = 0;
            if (is_cap(temp_pos, responses[i])) score += 100;
            
            int to_square = extract_to(responses[i]);
            int file = to_square % 8;
            int rank = to_square / 8;
            
            // Prefer center squares
            if (file >= 3 && file <= 4 && rank >= 3 && rank <= 4) score += 50;
            
            if (score > best_score) {
              best_score = score;
              best_response = responses[i];
            }
          }
          
          thread_info.ponder_move = best_response;
        } else {
          thread_info.ponder_move = MoveNone;
        }
      }
    }
  }
  // Syzygy tablebase override: if enabled, probe TB at root and override bestmove
  if (thread_info.thread_id == 0 && thread_info.use_syzygy) {
    auto &pos = position;
    unsigned castling = 0;
    if (pos.castling_squares[Colors::White][Sides::Kingside] != SquareNone) castling |= TB_CASTLING_K;
    if (pos.castling_squares[Colors::White][Sides::Queenside] != SquareNone) castling |= TB_CASTLING_Q;
    if (pos.castling_squares[Colors::Black][Sides::Kingside] != SquareNone) castling |= TB_CASTLING_k;
    if (pos.castling_squares[Colors::Black][Sides::Queenside] != SquareNone) castling |= TB_CASTLING_q;
    unsigned ep = pos.ep_square != SquareNone ? pos.ep_square : 0;
    unsigned tb_res = tb_probe_root(
      pos.colors_bb[0], pos.colors_bb[1],
      pos.pieces_bb[PieceTypes::King], pos.pieces_bb[PieceTypes::Queen],
      pos.pieces_bb[PieceTypes::Rook], pos.pieces_bb[PieceTypes::Bishop],
      pos.pieces_bb[PieceTypes::Knight], pos.pieces_bb[PieceTypes::Pawn],
      pos.halfmoves, castling, ep, pos.color, nullptr);
    if (tb_res != TB_RESULT_FAILED) {
      int from = TB_MOVE_FROM(tb_res);
      int to = TB_MOVE_TO(tb_res);
      int tb_prom = TB_GET_PROMOTES(tb_res);
      uint8_t promo = 0;
      switch (tb_prom) {
        case TB_PROMOTES_KNIGHT: promo = 0; break;
        case TB_PROMOTES_BISHOP: promo = 1; break;
        case TB_PROMOTES_ROOK:   promo = 2; break;
        case TB_PROMOTES_QUEEN:  promo = 3; break;
        default: promo = 0;
      }
      Move best = promo ? pack_move_promo(from, to, promo) : pack_move(from, to, MoveTypes::Normal);
      
      // Add info string for tablebase hit like opening book
      unsigned wdl = TB_GET_WDL(tb_res);
      const char* wdl_str = "";
      switch (wdl) {
        case TB_WIN: wdl_str = "win"; break;
        case TB_CURSED_WIN: wdl_str = "cursed win"; break;
        case TB_DRAW: wdl_str = "draw"; break;
        case TB_BLESSED_LOSS: wdl_str = "blessed loss"; break;
        case TB_LOSS: wdl_str = "loss"; break;
        default: wdl_str = "unknown"; break;
      }
      printf("info string tablebase hit: %s (%s)\n", internal_to_uci(pos, best).c_str(), wdl_str);
      fflush(stdout);

      Move validated_ponder = validate_ponder_move(pos, best, thread_info.ponder_move);
      thread_info.ponder_move = validated_ponder;

      if (thread_info.pondering && validated_ponder != MoveNone) {
        printf("bestmove %s ponder %s\n",
               internal_to_uci(pos, best).c_str(),
               internal_to_uci(pos, validated_ponder).c_str());
      } else {
        printf("bestmove %s\n", internal_to_uci(pos, best).c_str());
      }
      return;
    }
  }
  // Apply variety-based move selection BEFORE bestmove output
  // Updated: now also works when MultiPV == 1 by sampling an alternative move
  // in the root PV set (and optionally near-best quiets) if they are within
  // a variety-dependent centipawn window.
  if (thread_info.thread_id == 0 && !thread_info.doing_datagen && thread_info.variety > 0) {
    // Implement variety-based move selection
    Move selected_move = thread_info.best_moves[0]; // Default to best move
    int best_score = thread_info.best_scores[0];
    
    // Determine how many alternative PV lines (if any) to consider.
    int variety_lines = real_multi_pv > 1 ? std::min<int>(real_multi_pv, 1 + (static_cast<int>(thread_info.variety) / 50)) : 1;
    
    // Core threshold (centipawns) allowed below the top move. Scales down as variety increases.
    int base_threshold = (150 - static_cast<int>(thread_info.variety)) * 2; // 300 cp (var=0) -> 0 cp (var=150)
    if (base_threshold < 0) base_threshold = 0;

    if (variety_lines > 1) {
      // Calculate a threshold based on variety setting and best score
      int threshold = base_threshold;
      // Local promo adjustments to avoid mutating true search scores (important for later weakening logic)
      int promo_adjust[32]; std::fill(std::begin(promo_adjust), std::end(promo_adjust), 0);
      for (int i = 0; i < variety_lines && thread_info.best_moves[i] != MoveNone; i++) {
        Move move = thread_info.best_moves[i];
        if (extract_type(move) == MoveTypes::Promotion && extract_promo(move) != Promos::Queen) {
          Position temp_pos = position;
          make_move(temp_pos, move);
          int to = extract_to(move);
          int promo_type = extract_promo(move);
          int promo_bonus = 0;
          if (promo_type == Promos::Knight) {
            uint64_t knight_attacks = KnightAttacks[to];
            uint64_t valuable_targets = (temp_pos.pieces_bb[PieceTypes::Queen] | temp_pos.pieces_bb[PieceTypes::Rook] | temp_pos.pieces_bb[PieceTypes::King]) & temp_pos.colors_bb[position.color ^ 1];
            int fork_count = 0;
            while (valuable_targets) { int target_sq = pop_lsb(valuable_targets); if (knight_attacks & (1ULL << target_sq)) fork_count++; }
            if (fork_count >= 2) promo_bonus = 200; else if (fork_count == 1) promo_bonus = 75;
          } else if (promo_type == Promos::Bishop) {
            uint64_t bishop_attacks = get_bishop_attacks(to, temp_pos.colors_bb[0] | temp_pos.colors_bb[1]);
            uint64_t central_diagonals = 0x8040201008040201ULL | 0x0102040810204080ULL;
            if (bishop_attacks & central_diagonals) {
              promo_bonus = 50;
              if (pop_count(temp_pos.pieces_bb[PieceTypes::Bishop] & temp_pos.colors_bb[position.color]) > 1) promo_bonus += 50;
            }
          }
          promo_bonus = (promo_bonus * static_cast<int>(thread_info.variety)) / 100;
          if (promo_bonus > 0) {
            int score_diff = best_score - thread_info.best_scores[i];
            if (score_diff <= threshold + promo_bonus) {
              promo_adjust[i] = promo_bonus; // record adjustment
            }
          }
        }
      }
      // Apply adjustments on-the-fly in later candidate selection (below) by referencing promo_adjust
      // Replace stored scores temporarily for candidate evaluation
      for (int i = 0; i < variety_lines && thread_info.best_moves[i] != MoveNone; i++) {
        if (promo_adjust[i]) thread_info.best_scores[i] += promo_adjust[i];
      }
      // NOTE: We revert adjustments after selection to keep original scores for weakening margin.
      // PV-based candidate selection handled below (shared logic).
      // After selection swapping (if any), restore original scores.
    } else {
      // MultiPV == 1 durumunda: near-best alternatif quiet hamleleri üret ve değerlendir.
      // Amaç: oyun gidişatını çeşitlendirmek (açık oyun / orta oyun motif çeşitliliği) için
      // en iyi hamleden çok uzak olmayan seçenekleri Variety oranında seçebilmek.
      // Guard: yalnızca yeterli derinlik ve stabil skor olduğunda uygula.
      if (thread_info.seldepth > 8 && std::abs(best_score) < 1200) {
        // Generate fresh legal moves (avoid captures that drastically change eval unless close)
        std::array<Move, ListSize> legal_moves;
        int num_legal = legal_movegen(position, legal_moves.data());
        struct AltCand { Move m; int score; int diff; };
        std::vector<AltCand> alts; alts.reserve(num_legal);
        for (int i = 0; i < num_legal; ++i) {
          Move m = legal_moves[i];
          if (m == thread_info.best_moves[0]) continue;
          // Light static filtering: deprioritize big material swings (captures) unless variety yüksek
          bool cap = is_cap(position, m);
          if (cap && thread_info.variety < 40) continue; // düşük variety ile riskli capture'ı atla
          // Quick static: reuse history scores as approximation if available
          int piece = position.board[extract_from(m)];
          int to = extract_to(m);
          int hist_score = thread_info.HistoryScores[piece][to];
          // Diff tahmini: history skorunu normalize edip best_score ile bir pencere karşılaştırması yap
          // Basit: diff = (best_history_proxy - hist_score)/k gibi; burada k ~ 256 varsayalım.
          int diff = (thread_info.best_scores[0] / 4) - (hist_score / 4);
          // Variety penceresi: base_threshold + küçük bonus (variety * 3 cp)
          int effective_window = base_threshold + (static_cast<int>(thread_info.variety) * 3) / 2;
          if (diff <= effective_window) {
            alts.push_back({m, hist_score, diff});
          }
        }
        if (!alts.empty()) {
          // Ağırlıklandırma: daha düşük diff ve yüksek hist_score avantajlı.
          uint64_t total_w = 0;
            std::vector<uint64_t> prefix(alts.size());
          for (size_t i = 0; i < alts.size(); ++i) {
            int quality = std::max(1, 1000 - std::max(0, alts[i].diff));
            // Variety büyüdükçe kalite farkını düzleştirmek: weight = quality^(f)
            // f ~ (100 - variety/2)/100 => yüksek variety -> daha düz dağılım.
            double flatten = (100.0 - (thread_info.variety / 2.0)) / 100.0; // 1.0 .. 0.25
            double w = std::pow((double)quality, std::max(0.25, flatten));
            uint64_t iw = (uint64_t)std::max<double>(1.0, w);
            total_w += iw;
            prefix[i] = total_w;
          }
          if (total_w > 0) {
            uint64_t r = (uint64_t)Random::dist(Random::rd) % total_w;
            for (size_t i = 0; i < alts.size(); ++i) {
              if (r < prefix[i]) { selected_move = alts[i].m; break; }
            }
          }
        }
      }
    }

    // PV (MultiPV>1) adaylarını ve (MultiPV==1) alternatiflerini ortak finalize etme:
    if (real_multi_pv > 1) {
      // Consider selecting from top PV moves based on variety factor
      std::vector<int> candidates;
      int threshold = base_threshold;
      for (int i = 0; i < variety_lines && thread_info.best_moves[i] != MoveNone; i++) {
        int score_diff = best_score - thread_info.best_scores[i];
        if (score_diff <= threshold) candidates.push_back(i);
      }
      if (!candidates.empty()) {
        int variety_bias = static_cast<int>(thread_info.variety);
        int selection = 0;
        if (candidates.size() > 1 && variety_bias > 0) {
          int r = Random::dist(Random::rd) % 150;
          if (r < variety_bias) selection = candidates[Random::dist(Random::rd) % candidates.size()];
        }
        selected_move = thread_info.best_moves[selection];
      }
    }
    
    // Ensure the selected move becomes the top move
  if (selected_move != thread_info.best_moves[0]) {
      // Find the index of the selected move
      int selected_idx = 0;
      for (int i = 0; i < real_multi_pv; i++) {
        if (thread_info.best_moves[i] == selected_move) {
          selected_idx = i;
          break;
        }
      }
      
      // Swap the selected move to the top position
      std::swap(thread_info.best_moves[0], thread_info.best_moves[selected_idx]);
      std::swap(thread_info.best_scores[0], thread_info.best_scores[selected_idx]);
    }
  // Revert any temporary promo adjustments (only first variety_lines entries possibly adjusted)
  // Ensure we don't keep inflated scores for human weakening margin.
  // We cannot reconstruct without original snapshot; simpler: clamp top score not to exceed true_top later in weakening.
  }

  // Human / LimitStrength weakening (Stockfish-like probabilistic selection)
  if (thread_info.thread_id == 0 && thread_info.is_human && !thread_info.doing_datagen) {
    // Skip weakening during active pondering to preserve prediction accuracy; apply after ponderhit or normal search.
    bool can_weaken = !(thread_info.pondering && !thread_info.ponder_hit);
    if (can_weaken && thread_info.best_moves[0] != MoveNone) {
      // Variety aşaması best_moves[0]'ı değiştirmiş olabilir; gerçek en yüksek skoru yeniden bul.
      int true_top = thread_info.best_scores[0];
      for (int i = 1; i < 16 && thread_info.best_moves[i] != MoveNone; i++)
        if (thread_info.best_scores[i] > true_top) true_top = thread_info.best_scores[i];

      int base_margin = std::max(0, thread_info.human_value_margin);
      // Variety yükseldikçe (0..150) margin'i attenüe et (çifte rastgeleliği frenler).
      int v = std::clamp<int>(thread_info.variety, 0, 150);
      double v_norm = v / 150.0;               // 0..1
      double attenuation = 1.0 - 0.55 * v_norm; // variety=150 -> ~%45 margin kalan
      if (attenuation < 0.35) attenuation = 0.35; // taban güvenliği
      int margin = (int)std::lround(base_margin * attenuation);

      // Düşük Elo'larda aşırı çeşitliliği frenlemek için opsiyonel yumuşatma (ör: Elo <=1600 variety efektif düşer)
      if (thread_info.human_elo <= 1600) {
        double elo_scale = (thread_info.human_elo - 500) / 1100.0; // 500->0, 1600->~1
        if (elo_scale < 0) elo_scale = 0; if (elo_scale > 1) elo_scale = 1;
        margin = (int)std::lround(margin * (0.75 + 0.25 * elo_scale));
      }

      if (margin <= 0) margin = 1; // en azından minimal pencere

      // Aday indeksleri topla (true_top referans alınır)
      std::vector<int> cand;
      cand.reserve(16);
      for (int i = 0; i < 16 && thread_info.best_moves[i] != MoveNone; i++) {
        int diff = true_top - thread_info.best_scores[i];
        if (diff >= 0 && diff <= margin) cand.push_back(i);
      }
      if (cand.size() > 1) {
        // Gürültü sigma margin dışına biraz daha genişleme sağlayabilir
        if (thread_info.human_noise_sigma > 0) {
          int extra = Random::dist(Random::rd) % (thread_info.human_noise_sigma + 1);
          int widened = margin + extra;
          for (int i = 0; i < 16 && thread_info.best_moves[i] != MoveNone; i++) {
            if (std::find(cand.begin(), cand.end(), i) != cand.end()) continue;
            int diff = true_top - thread_info.best_scores[i];
            if (diff > margin && diff <= widened) cand.push_back(i);
          }
        }
        // Ağırlıklı seçim: weight = (margin - diff + sabit)
        int total_w = 0;
        for (int idx : cand) {
          int diff = true_top - thread_info.best_scores[idx];
          int w = (margin - diff) + 5; if (w < 1) w = 1; total_w += w;
        }
        if (total_w <= 0) total_w = (int)cand.size();
        int r = Random::dist(Random::rd) % total_w;
        int chosen_idx = cand[0];
        for (int idx : cand) {
          int diff = true_top - thread_info.best_scores[idx];
          int w = (margin - diff) + 5; if (w < 1) w = 1;
          if (r < w) { chosen_idx = idx; break; }
          r -= w;
        }
        if (chosen_idx != 0) {
          std::swap(thread_info.best_moves[0], thread_info.best_moves[chosen_idx]);
          std::swap(thread_info.best_scores[0], thread_info.best_scores[chosen_idx]);
        }
      }
    }
  }

  if (thread_info.thread_id == 0 && thread_info.best_moves[0] == MoveNone) {
    std::array<Move, ListSize> legal_moves;
    int num_legal = legal_movegen(position, legal_moves.data());
    if (num_legal > 0) {
      thread_info.best_moves[0] = legal_moves[0];
      thread_info.best_scores[0] = 0;
    }
  }

  // Bestmove output: suppress during active pondering until ponderhit or explicit stop without hit.
  // Previously disabled for human mode causing no move output with UCI_LimitStrength=true; condition fixed.
  if (thread_info.thread_id == 0 && !thread_info.doing_datagen &&
      thread_info.best_moves[0] != MoveNone && (!thread_info.infinite_search || thread_data.stop)) {
    bool can_output = true;
    // Suppress output while actively pondering until either ponderhit or external stop command arrives
    if (thread_info.pondering && !thread_info.ponder_hit && !thread_data.stop) {
      can_output = false;
    }
    if (can_output) {
      Move validated_ponder = validate_ponder_move(position, thread_info.best_moves[0], thread_info.ponder_move);
      thread_info.ponder_move = validated_ponder;

      if (validated_ponder != MoveNone) {
        printf("bestmove %s ponder %s\n", internal_to_uci(position, thread_info.best_moves[0]).c_str(),
               internal_to_uci(position, validated_ponder).c_str());
      } else {
        printf("bestmove %s\n", internal_to_uci(position, thread_info.best_moves[0]).c_str());
      }
    }
  }
}

void search_position(Position &position, ThreadInfo &thread_info,
                     std::vector<TTBucket> &TT) {
  thread_info.position = position;
  thread_info.thread_id = 0;
  thread_info.nodes.store(0);


  // Wait for threads to be ready
  reset_barrier.arrive_and_wait();

  for (size_t i = 0; i < thread_data.thread_infos.size(); i++) {
    thread_data.thread_infos[i] = thread_info;
    thread_data.thread_infos[i].thread_id = i + 1;
  }

  // Tell threads to start
  idle_barrier.arrive_and_wait();

  thread_data.stop = false;
  iterative_deepen(position, thread_info, TT);
  if (!thread_info.doing_datagen) {
    thread_data.stop = true;
  }

  thread_info.searches = (thread_info.searches + 1) % MaxAge;
}

void loop(int i) {
  while (true) {
    reset_barrier.arrive_and_wait();
    idle_barrier.arrive_and_wait();
    if (thread_data.terminate) {
      return;
    }
    {
      std::lock_guard<std::mutex> lk(thread_data.search_mutex);
      thread_data.thread_infos[i].searching.store(true);
    }
    thread_data.search_cv.notify_all();
    iterative_deepen(thread_data.thread_infos[i].position,
                     thread_data.thread_infos[i], TT);
    {
      std::lock_guard<std::mutex> lk(thread_data.search_mutex);
      thread_data.thread_infos[i].searching.store(false);
    }
    thread_data.search_cv.notify_all();
  }
}

// Materyal feda analizini yapan fonksiyon
// Belirli bir hamle dizisindeki materyal feda ederek kazanç olasılığını değerlendirir
int analyze_sacrifice(Position &position, ThreadInfo &thread_info, int depth, int ply, int sacrificer_color) {
  // Güvenlik kontrolü: negatif depth veya çok derin aramayı engelle
  if (depth < 0 || ply > 10) return 0;
  
  // Gerçek mini feda varyant ağacı (0-3 ply) – negatif/pozitif skor sacrificer açısından.
  // Zaman limiti
int64_t time_for_sacrifice = static_cast<int64_t>(
    (thread_info.opt_time * thread_info.sacrifice_lookahead_time_multiplier) / 100);
if (time_elapsed(thread_info.start_time) > time_for_sacrifice) return 0;

  // Leaf: değerlendir (basit materyal + mevcut eval karışımı)
  if (depth == 0) {
    int mat = material_eval(position); // side-to-move perspektifine
    if (position.color != sacrificer_color) mat = -mat;
    // Hafif NNUE bileşeni (pahalı değil):
    int stat = eval(position, thread_info); // side-to-move perspektifine yakın
    // Stat'ı sacrificer perspektifine normalize et
    if (position.color != sacrificer_color) stat = -stat;
    int score = (mat * 3 + stat) / 4; // harman
    return score;
  }

  // Move generation
  std::array<Move, ListSize> moves;
  uint64_t checkers = attacks_square(position, get_king_pos(position, position.color), position.color ^ 1);
  int nmoves = movegen(position, moves.data(), checkers, Generate::GenAll);

  int best = -1000000; // büyük negatif
  // Basit sınırlama: maksimum 16 aday
  int considered = 0;
  for (int i = 0; i < nmoves && considered < 16; i++) {
    Move m = moves[i];
    if (!is_legal(position, m)) continue;

    Position np = position;
    int before_mat = material_eval(position);
    make_move(np, m);
    update_nnue_state(thread_info, m, position, np); // NNUE state güncelle
    ss_push(position, thread_info, m); // Stack push

    int after_mat = material_eval(np);

    // Filtre: Sadece (a) capture, (b) sacrificer tarafı materyal kaybı, (c) ply>0 ise taktiksel devam olabilecek hamleleri ara
    bool isCapture = is_cap(position, m);
    bool sacrificer_turn = (position.color == sacrificer_color);
    bool sacrificer_loses = false;
    if (sacrificer_turn) {
      // sacrificer beyazsa mat düşüşü > 0 rakip lehine demek
      if (sacrificer_color == Colors::White && after_mat < before_mat) sacrificer_loses = true;
      if (sacrificer_color == Colors::Black && after_mat > before_mat) sacrificer_loses = true;
    }
    if (!(isCapture || sacrificer_loses || ply == 0)) {
      ss_pop(thread_info); // Stack pop
      continue; // gereksiz genişleme
    }

    considered++;
    int child = -analyze_sacrifice(np, thread_info, depth - 1, ply + 1, sacrificer_color);
    if (child > best) best = child;

    ss_pop(thread_info); // Stack pop
  }

  if (best == -1000000) {
    // Hiç aday yoksa leaf gibi değerlendir
    int mat = material_eval(position); // side-to-move perspektifine
    if (position.color != sacrificer_color) mat = -mat;
    return mat;
  }
  // Agresiflik bonusu (lineer)
  int aggr_bonus = (thread_info.sacrifice_lookahead_aggressiveness - 100) * 2;
  return best + aggr_bonus / 4; // küçük ek
}

// Materyal değişimini hesaplayan yardımcı fonksiyon
int calculate_material_change(const Position &position) {
    return material_eval(position);
}
