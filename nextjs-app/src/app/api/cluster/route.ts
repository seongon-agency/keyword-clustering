import { NextRequest, NextResponse } from "next/server";

// Extend timeout to 5 minutes for long clustering operations
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch("http://localhost:8000/cluster", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      // No timeout - let it run as long as needed
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Cluster API error:", error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Failed to connect to backend" },
      { status: 500 }
    );
  }
}
