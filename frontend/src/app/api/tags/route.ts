export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const astar_URL =
    process.env.NEXT_PUBLIC_astar_URL || "http://localhost:11434";
  const res = await fetch(astar_URL + "/api/tags");
  return new Response(res.body, res);
}
